#!/usr/bin/env python3
"""
Trains a Multi-Token Prediction (MTP) head on top of a frozen base model.
The MTP head predicts multiple future tokens from the current hidden state,
accelerating inference through speculative decoding or parallel prediction.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM


class JsonlTokenDataset(IterableDataset):
    """
    An iterable dataset that reads tokenized examples from sharded JSONL files.
    """
    def __init__(self, data_dir: Path, max_length: int):
        self.files = sorted(data_dir.glob("shard_*.jsonl"))
        self.max_length = max_length
        if not self.files:
            raise FileNotFoundError(f"No shard_*.jsonl files found in {data_dir}")

    def __iter__(self):
        for path in self.files:
            with path.open() as f:
                for line in f:
                    ids = json.loads(line)["input_ids"][: self.max_length]
                    if len(ids) >= 16:
                        yield torch.tensor(ids, dtype=torch.long)


def collate(batch):
    """
    Pads sequences in a batch to the maximum length.
    Returns input_ids and attention_mask.
    """
    max_len = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head.
    Consists of a sequence of residual-like blocks that transform the
    base model's hidden states to predict future tokens.
    """
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                RMSNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(depth)
        )

    def forward(self, hidden_states):
        """
        Returns a list of predicted hidden states, one for each prediction depth.
        """
        return [block(hidden_states) for block in self.blocks]


def pick_lm_weight(model):
    """
    Extracts the language modeling head weights (output embeddings) from the model.
    Used for calculating logits from the MTP head's hidden states.
    """
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings():
        return model.get_output_embeddings().weight
    if hasattr(model, "get_input_embeddings") and model.get_input_embeddings():
        return model.get_input_embeddings().weight
    raise RuntimeError("Could not find tied lm/input embedding weight")


def main():
    parser = argparse.ArgumentParser(description="Train an MTP head on a frozen base model.")
    parser.add_argument("--model", required=True, help="Base model to use.")
    parser.add_argument("--data", required=True, type=Path, help="Directory containing tokenized shards.")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the trained MTP head.")
    parser.add_argument("--mtp-depth", type=int, default=2, help="Number of future tokens to predict.")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max-steps", type=int, help="Limit total training steps.")
    args = parser.parse_args()

    # Device and precision setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # Load base model and freeze it
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Initialize MTP head and optimizer
    hidden_size = model.config.hidden_size
    head = MTPHead(hidden_size, args.mtp_depth).to(device=device, dtype=dtype)
    lm_weight = pick_lm_weight(model).detach()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)

    dataset = JsonlTokenDataset(args.data, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}")
        for micro_step, (input_ids, attention_mask) in enumerate(pbar, start=1):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get hidden states from the base model
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden = out.hidden_states[-1]

            total_loss = None
            valid_losses = 0
            preds = head(hidden)
            
            # Calculate weighted loss for each prediction depth
            for depth_idx, pred_hidden in enumerate(preds, start=1):
                # We can only predict up to the end of the sequence
                if pred_hidden.shape[1] <= depth_idx:
                    continue
                
                # Predict future tokens using the shared LM head weight
                logits = F.linear(pred_hidden[:, :-depth_idx, :], lm_weight)
                labels = input_ids[:, depth_idx:]
                mask = attention_mask[:, depth_idx:].bool()
                
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="none",
                ).reshape_as(labels)
                
                loss = loss[mask].mean()
                
                # Discount loss for further-out predictions
                weight = math.pow(0.8, depth_idx - 1)
                total_loss = loss * weight if total_loss is None else total_loss + loss * weight
                valid_losses += 1

            if total_loss is None or valid_losses == 0:
                continue

            # Normalize loss and backpropagate
            total_loss = total_loss / valid_losses / args.grad_accum
            total_loss.backward()

            # Optimizer step and logging
            if micro_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                pbar.set_postfix(loss=f"{total_loss.item() * args.grad_accum:.4f}")
                if args.max_steps and global_step >= args.max_steps:
                    break
        if args.max_steps and global_step >= args.max_steps:
            break

    # Save the trained MTP head tensors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {f"mtp_head.{k}": v.detach().cpu() for k, v in head.state_dict().items()}
    save_file(state, args.output)
    print(f"Saved MTP head to {args.output}")


if __name__ == "__main__":
    main()
