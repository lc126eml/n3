# Repository Guidelines

## Project Structure & Module Organization

This is a Python research/training codebase. Core entry points are in `training/`: `launch.py` starts Hydra training, `eval.py` evaluates a saved run, and `trainer.py` contains the main loop. Model code is split between `training/vggt/` and `training/vggt_omega/`; dataset loaders are in `training/datasets/`; shared utilities are in `training/train_utils/`; metrics and alignment helpers are in `training/eval_utils/`. YAML configs live under `training/configs/`, with experiments in `training/configs/experiment/`. Kaggle automation is in `kaggle/`. `transformers/` and `dust3r/` are local source trees. Notes are under `notes/`.

## Build, Test, and Development Commands

- `cd training && torchrun --nproc_per_node=4 launch.py`: run distributed training with the default config.
- `cd training && python launch.py --cfg vggt`: run a single-process training job with `training/configs/vggt.yaml`.
- `cd training && python launch.py --cfg experiment/align_ablation_first_frame.yaml`: run a specific experiment config.
- `cd training && python eval.py --logdir logs/<exp>/<run> --data_cfg configs/data/standalone_multiview_train.yaml`: evaluate a checkpointed run.
- `PYTHONPATH=training python -m py_compile training/trainer.py training/vggt_omega/models/vggt_omega.py`: quick syntax check for touched modules.

There is no project-level build system or declared test suite in this checkout.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation. Follow existing naming: snake_case for functions, variables, config keys, and file names; PascalCase for classes such as `Trainer`; uppercase only for constants. Prefer typed signatures where nearby code uses them. Keep Hydra config names descriptive and lowercase, for example `pts_align_to_gt.yaml` or `model_scaling_large.yaml`. Avoid unrelated formatting churn in local dependency trees.

## Testing Guidelines

Add focused checks near the code you change. For model or loss changes, run a small import/shape smoke test when feasible and at least `py_compile` on affected files. For dataset and geometry utilities, validate tensor/image shapes and camera intrinsics on a tiny synthetic sample. Do not commit generated logs, checkpoints, TensorBoard runs, or notebook outputs; `.gitignore` excludes `logs/`, `output/`, `*.pt`, `*.ckpt`, and `*.ipynb`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or noun-phrase subjects such as `global registers`, `max_scale in random_crop`, and `skip optimizer on nonfinite`. Keep the first line concise and specific. PRs should describe the training or evaluation behavior changed, list configs touched, mention datasets/checkpoints needed to reproduce, and include key command outputs or metrics. Link related issues or experiment notes when applicable.

## Security & Configuration Tips

Treat `kaggle/tokens.yaml`, dataset paths, and checkpoint locations as local secrets or machine-specific state. Do not add real credentials, private data paths, or large model weights to version control.
