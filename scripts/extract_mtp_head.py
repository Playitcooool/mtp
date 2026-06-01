#!/usr/bin/env python3
"""
Extracts Multi-Token Prediction (MTP) head tensors from a safetensors model checkpoint.
This is useful for isolating the MTP head for analysis or deployment.
"""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file


def collect_safetensors(model_dir: Path) -> list[Path]:
    """
    Identifies all safetensors files in a model directory.
    If a model.safetensors.index.json exists, it uses it to find shards.
    Otherwise, it lists all .safetensors files in the directory.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        names = sorted(set(index.get("weight_map", {}).values()))
        return [model_dir / name for name in names]

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract tensors whose names contain 'mtp' from a safetensors checkpoint."
    )
    parser.add_argument("model_dir", type=Path, help="Directory containing the model checkpoints.")
    parser.add_argument("output", type=Path, help="Path to save the extracted MTP tensors.")
    parser.add_argument("--prefix", default="mtp", help="Case-insensitive tensor-name filter (default: 'mtp').")
    args = parser.parse_args()

    tensors = {}
    source_files = collect_safetensors(args.model_dir)
    needle = args.prefix.lower()

    # Iterate through all shards and extract matching tensors
    for shard in source_files:
        with safe_open(shard, framework="np") as f:
            for key in f.keys():
                if needle in key.lower():
                    tensors[key] = f.get_tensor(key)

    if not tensors:
        print(f"No tensors matching {args.prefix!r} found in {args.model_dir}")
        print("This checkpoint likely does not include the MTP head.")
        return

    # Create parent directory if it doesn't exist and save the tensors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    print(f"Saved {len(tensors)} tensors to {args.output}")
    for key in sorted(tensors):
        print(key)


if __name__ == "__main__":
    main()
