# Qwen3.5 MTP Head Training

This repo trains a small multi-token prediction head for Qwen3.5-2B using
UltraChat. The base model is frozen; only lightweight future-token projection
blocks are trained.

## Why not train a full draft model?

For a 2B target model on a MacBook Air M4 24GB, a separate 0.8B draft model is
often too expensive to be useful. A small MTP head is the better first attempt:
it adds little memory, reuses the target model hidden states, and predicts the
next few tokens for speculative verification.

## Install

```bash
rtk pip install torch safetensors tqdm datasets transformers accelerate
```

For Apple Silicon training, PyTorch MPS is used automatically when available.

## Prepare UltraChat

This streams the full UltraChat dataset and writes tokenized training shards.

```bash
rtk python scripts/prepare_ultrachat.py \
  --model Qwen/Qwen3.5-2B \
  --output data/ultrachat_qwen35 \
  --max-length 1024 \
  --shard-size 20000
```

Use `--limit 20000` for a quick smoke test before running the full dataset.

## Train

```bash
rtk python scripts/train_mtp_head.py \
  --model Qwen/Qwen3.5-2B \
  --data data/ultrachat_qwen35 \
  --output artifacts/qwen35_2b_mtp_head.safetensors \
  --mtp-depth 2 \
  --max-length 1024 \
  --batch-size 1 \
  --grad-accum 16 \
  --lr 1e-4 \
  --epochs 1
```

The script saves only the trainable MTP head weights. The base model remains
unchanged.

## Practical Recommendation

Start with:

- `--mtp-depth 2`
- `--max-length 1024`
- `--limit 20000` for a first run
- full UltraChat only after the smoke test loss decreases

Full UltraChat on an Air is possible, but it is a long run. Expect many hours
to multiple days depending on thermal throttling and sequence length.
