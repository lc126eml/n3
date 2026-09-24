import argparse
import os

from hydra import initialize, compose
from trainer import Trainer
from train_utils.distributed import run_distributed


import rootutils
import torch

project_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["PROJECT_ROOT"] = str(project_root)


def main():
    parser = argparse.ArgumentParser(description="Run VGGT-Omega training.")
    parser.add_argument('--cfg', type=str, default='default', help='Name of the config file to use (without .yaml extension).')
    args = parser.parse_args()

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=args.cfg)

    if not cfg.get("distributed", {}).get("enabled", False)  or torch.cuda.device_count() <= 1:
        Trainer(cfg).run()
        return

    run_distributed(cfg)


if __name__ == "__main__":
    main()
# python launch.py --cfg default
# python launch.py --cfg vggt
# python launch.py --cfg eval
# python launch.py --cfg experiment/align_ablation_first_frame.yaml
