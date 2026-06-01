#!/usr/bin/env python3
"""
Prepares the UltraChat dataset for MTP training.
This script downloads the dataset, tokenizes it using a specified model's tokenizer,
and saves the result into sharded JSONL files.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def normalize_messages(example):
    """
    Normalizes different UltraChat dataset formats into a standard list of messages.
    Supports formats with 'messages' or 'data' keys.
    """
    if "messages" in example and example["messages"]:
        return example["messages"]
    if "data" in example and example["data"]:
        roles = ["user", "assistant"]
        return [
            {"role": roles[i % 2], "content": text}
            for i, text in enumerate(example["data"])
            if text
        ]
    return []


def render_chat(tokenizer, messages):
    """
    Renders a list of messages into a single string using the tokenizer's chat template.
    Falls back to a simple 'role: content' format if the template fails.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def write_shard(output_dir, shard_id, rows):
    """
    Writes a list of tokenized examples to a JSONL shard file.
    """
    path = output_dir / f"shard_{shard_id:05d}.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Prepare UltraChat dataset for MTP training.")
    parser.add_argument("--model", required=True, help="Path or name of the model whose tokenizer to use.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to save the processed shards.")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k", help="HF dataset name.")
    parser.add_argument("--split", default="train_sft", help="Dataset split to use.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--shard-size", type=int, default=20000, help="Number of examples per shard.")
    parser.add_argument("--limit", type=int, help="Optional limit on total examples.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)

    rows = []
    shard_id = 0
    total = 0
    for example in tqdm(dataset, desc="tokenizing"):
        messages = normalize_messages(example)
        if not messages:
            continue
        
        text = render_chat(tokenizer, messages)
        # Tokenize and truncate to max length
        ids = tokenizer(text, add_special_tokens=False).input_ids[: args.max_length]
        
        # Filter out very short sequences
        if len(ids) < 16:
            continue
            
        rows.append({"input_ids": ids})
        total += 1
        
        # Write shard when shard size is reached
        if len(rows) >= args.shard_size:
            write_shard(args.output, shard_id, rows)
            shard_id += 1
            rows = []
        
        if args.limit and total >= args.limit:
            break

    # Write remaining rows
    if rows:
        write_shard(args.output, shard_id, rows)

    # Save metadata about the processing run
    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "max_length": args.max_length,
        "examples": total,
    }
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {total} examples to {args.output}")


if __name__ == "__main__":
    main()
