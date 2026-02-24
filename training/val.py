from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
try:
    from filelock import FileLock
except ImportError:
    FileLock = None

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="eval")      # loads default.yaml

# --- Acquire a file lock to ensure exclsive GPU usage ---
if FileLock:
    lock_path = "/tmp/gpu.lock"
    gpu_lock = FileLock(lock_path)
    print(f"Attempting to acquire lock on '{lock_path}'...")
    gpu_lock.acquire()
    print("Lock acquired. It is safe to proceed.")
    # The lock will be automatically released when the script exits.
else:
    print("`filelock` library not found, skipping lock. Run `pip install filelock`.")

trainer = Trainer(cfg)
trainer.run()
m=1
