# AGENTS.md

## Environment

- Use `conda run -n sepfp` for code execution, validation, and training-related commands unless explicitly told otherwise.

## Hardware

- `2x 1080 Ti (11GB)`; VRAM is limited.
- Prefer `fp16`, small batch sizes.

## Compute

- CPU is limited; keep `num_workers <= 6`.
- Avoid heavy on-the-fly preprocessing during training.
