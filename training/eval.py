import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import torch

import rootutils

# Same root setup as launch.py
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def find_trainer_config(logdir: str) -> Path:
    """
    Find the first 'trainer_config.yaml' under logdir and return its path.
    Used only to determine that logdir is a valid experiment directory.
    """
    logdir_path = Path(logdir)
    for p in logdir_path.rglob("trainer_config.yaml"):
        return p
    raise FileNotFoundError(f"No 'trainer_config.yaml' found under: {logdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VGGT evaluation.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Training log directory containing ckpts/ and trainer_config.yaml.",
    )
    parser.add_argument(
        "--data_cfg",
        type=str,
        default=None,
        help="Optional extra data config YAML file. "
             "If provided, its values override the corresponding fields in the main cfg.",
    )
    args = parser.parse_args()

    logdir = Path(args.logdir).resolve()

    # Ensure logdir is valid and obtain the subdir containing trainer_config.yaml
    trainer_cfg_path = find_trainer_config(str(logdir))
    config_dir = str(trainer_cfg_path.parent)   # directory containing trainer_config.yaml
    config_name = trainer_cfg_path.stem         # usually 'trainer_config'

    # Expected checkpoint path: {logdir}/ckpts/checkpoint.pt
    checkpoint_path = logdir / "ckpts" / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Use initialize_config_dir for an arbitrary (absolute) config directory
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg: DictConfig = compose(
            config_name=config_name,
            overrides=[
                "mode=val",
                f"checkpoint.resume_checkpoint_path={str(checkpoint_path)}",
                f"checkpoint.strict=True",
                f"checkpoint.filter_keys.default_keep=True",
            ],
        )

    # Optional: load extra data config and override corresponding fields in cfg
    if args.data_cfg is not None:
        data_cfg_path = Path(args.data_cfg).resolve()
        if not data_cfg_path.is_file():
            raise FileNotFoundError(f"Data config file not found: {data_cfg_path}")
        data_cfg = OmegaConf.load(str(data_cfg_path))
        # Later arguments override earlier ones, so data_cfg values take precedence.
        cfg.data = OmegaConf.merge(cfg.data, data_cfg)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    # cfg.logging.log_freq = 1
    trainer = Trainer(cfg)
    trainer.run()
# python eval.py --logdir /lc/code/3D/vggt-training/training/logs/exp002/2025-12-03_19-59-44 --data_cfg /lc/code/3D/vggt-training/training/configs/data/standalone_multiview_train.yaml
