# AGENTS.md

## Environment

- Use `conda run -n sepfp` for code execution, validation, and training-related commands unless explicitly told otherwise.

## Hardware

- `2x 1080 Ti (11GB)`; VRAM is limited.
- Prefer `fp16`, small batch sizes.

## Compute

- CPU is limited; keep `num_workers <= 6`.
- Avoid heavy on-the-fly preprocessing during training.

## Git

- Remote: `origin` -> `https://github.com/woojinkang01/SepFP.git`
- Default branch: `main`
- Keep datasets, checkpoints, logs, wandb runs, local envs, and experiment outputs out of Git.
- Do not commit, push, branch, rewrite history, or change remotes unless explicitly requested.
- Before committing, check `git status --short` and staged changes.
