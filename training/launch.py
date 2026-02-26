import argparse
import os
from hydra import initialize, compose
from trainer import Trainer


import rootutils

project_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ["PROJECT_ROOT"] = str(project_root)

parser = argparse.ArgumentParser(description="Run VGGT training.")
parser.add_argument('--config_name', type=str, default='default', help='Name of the config file to use (without .yaml extension).')
args = parser.parse_args()

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name=args.config_name)

trainer = Trainer(cfg)
trainer.run()
# python launch.py --config_name vggt
# python launch.py --config_name eval
