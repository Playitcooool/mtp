#!/usr/bin/env python3
"""
Train a small MTP head on top of a frozen MLX language model.

The base model is used only to produce hidden states. Gradients are computed
for the MTP head parameters only.
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.utils import load


class JsonlTokenDataset:
    def __init__(self, data_dir: Path, max_length: int):
        self.files = sorted(data_dir.glob("shard_*.jsonl"))
        self.max_length = max_length
        if not self.files:
            raise FileNotFoundError(f"No shard_*.jsonl files found in {data_dir}")

    def __iter__(self):
        while True:
            for path in self.files:
                with path.open() as f:
                    for line in f:
                        ids = json.loads(line)["input_ids"][: self.max_length]
                        if len(ids) >= 16:
                            yield ids


def batch_iter(dataset, batch_size: int):
    batch = []
    for ids in dataset:
        batch.append(ids)
        if len(batch) == batch_size:
            lengths = np.array([len(x) for x in batch], dtype=np.int32)
            max_len = int(lengths.max())
            input_ids = np.zeros((batch_size, max_len), dtype=np.int32)
            for i, row in enumerate(batch):
                input_ids[i, : len(row)] = row
            yield mx.array(input_ids), mx.array(lengths)
            batch = []


class MTPHead(nn.Module):
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.blocks = [
            nn.Sequential(
                nn.RMSNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(depth)
        ]

    def __call__(self, hidden_states):
        return [block(hidden_states) for block in self.blocks]


def lm_project(model, hidden_states):
    lm = model.language_model
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(hidden_states)
    return lm.lm_head(hidden_states)


def mtp_loss(head, hidden, input_ids, lengths, base_model, depth: int):
    total = mx.array(0.0, dtype=mx.float32)
    valid_losses = 0
    preds = head(hidden)

    for depth_idx, pred_hidden in enumerate(preds, start=1):
        if pred_hidden.shape[1] <= depth_idx:
            continue

        logits = lm_project(base_model, pred_hidden[:, :-depth_idx, :])
        labels = input_ids[:, depth_idx:]
        positions = mx.arange(labels.shape[1])[None, :] + depth_idx
        mask = positions < lengths[:, None]

        losses = nn.losses.cross_entropy(logits, labels).astype(mx.float32)
        denom = mx.maximum(mask.sum(), 1)
        loss = (losses * mask).sum() / denom
        total = total + loss * math.pow(0.8, depth_idx - 1)
        valid_losses += 1

    if valid_losses == 0:
        return total
    return total / valid_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-save", type=int, default=100)
    args = parser.parse_args()

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    print(f"Loading MLX model from {args.model}", flush=True)
    model, _ = load(args.model)
    model.eval()
    model.freeze()
    hidden_size = model.args.text_config["hidden_size"]

    head = MTPHead(hidden_size, args.mtp_depth)
    head_dtype = {
        "float32": mx.float32,
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
    }[args.head_dtype]
    head.set_dtype(head_dtype)

    # Use Muon for 2D weights (Linear layers) and AdamW for 1D (RMSNorm, biases)
    muon = optim.Muon(learning_rate=args.lr)
    adamw = optim.AdamW(learning_rate=args.lr)

    def select_optimizer(path, value):
        return muon if value.ndim == 2 else adamw

    optimizer = optim.MultiOptimizer(select_optimizer)
    loss_and_grad = nn.value_and_grad(head, mtp_loss)

    dataset = JsonlTokenDataset(args.data, args.max_length)
    batches = batch_iter(iter(dataset), args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting MLX MTP training: max_steps={args.max_steps}, "
        f"max_length={args.max_length}, grad_accum={args.grad_accum}",
        flush=True,
    )

    accum_grads = None
    accum_loss = 0.0
    accum_count = 0
    tic = time.perf_counter()

    for micro_step, (input_ids, lengths) in enumerate(batches, start=1):
        hidden = model.language_model.model(input_ids)
        hidden = hidden.astype(head_dtype)
        mx.eval(hidden)

        loss, grads = loss_and_grad(
            head,
            hidden,
            input_ids,
            lengths,
            model,
            args.mtp_depth,
        )
        loss_value = loss.item()
        accum_loss += loss_value
        accum_count += 1

        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = tree_map(lambda x, y: x + y, accum_grads, grads)

        if micro_step % args.grad_accum == 0:
            step = micro_step // args.grad_accum
            scaled_grads = tree_map(lambda x: x / args.grad_accum, accum_grads)
            optimizer.update(head, scaled_grads)
            mx.eval(head.parameters(), optimizer.state)
            mx.clear_cache()
            accum_grads = None

            if step % args.steps_per_report == 0 or step == 1:
                elapsed = time.perf_counter() - tic
                print(
                    f"Step {step}: loss={accum_loss / accum_count:.4f}, "
                    f"micro_steps={micro_step}, "
                    f"sec/step={elapsed / max(step, 1):.2f}, "
                    f"peak_mem={mx.get_peak_memory() / 1e9:.2f} GB",
                    flush=True,
                )

            if step % args.steps_per_save == 0 or step == args.max_steps:
                state = {
                    f"mtp_head.{k}": v
                    for k, v in tree_flatten(head.trainable_parameters())
                }
                mx.save_safetensors(str(args.output), state)
                print(f"Step {step}: saved {args.output}", flush=True)

            if step >= args.max_steps:
                break

    state = {
        f"mtp_head.{k}": v
        for k, v in tree_flatten(head.trainable_parameters())
    }
    mx.save_safetensors(str(args.output), state)
    print(f"Saved final MTP head to {args.output}", flush=True)


if __name__ == "__main__":
    main()
