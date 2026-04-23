from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from training.datasets_tpu import get_data_loader
from training.datasets_tpu import *


class StandaloneMultiViewDataModuleTPU:
    """
    TPU-only datamodule variant.

    It mirrors the standalone datamodule but routes all dataloaders through the
    TPU-aware loader entrypoint so world_size/rank sharding stays isolated from
    the default GPU path.
    """

    def __init__(
        self,
        train_config: dict,
        validation_config: dict,
        test_config: dict,
        pin_memory: bool,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        **kwargs,
    ) -> None:
        self.train_config = train_config
        self.validation_config = validation_config
        self.test_config = test_config
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.accum_steps = int(kwargs.get("accum_steps", 1))
        self.prefork_train_dataset = kwargs.get("prefork_train_dataset")

    def _loader_kwargs(self, config: dict) -> Dict[str, Any]:
        return dict(
            num_workers=config["num_workers"],
            pin_mem=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            seed=self.seed,
            dynamic_schedule=config.get("dynamic_schedule", "batch"),
            world_size=self.world_size,
            rank=self.rank,
            distributed_sampler_mode=config.get("distributed_sampler_mode", "drop_last"),
        )

    def _dataloader_from_cfg(self, config: dict):
        if not config.get("datasets") or not all(isinstance(d, str) for d in config["datasets"]):
            raise ValueError("All datasets must be provided as a list of strings.")

        try:
            datasets = [eval(dataset) for dataset in config["datasets"]]
        except NameError as exc:
            print(
                f"Error during eval: {exc}. Make sure dataset classes and transforms are imported.",
                flush=True,
            )
            raise

        val_loaders = []
        for dataset in datasets:
            loader = get_data_loader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                drop_last=False,
                **self._loader_kwargs(config),
            )
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)
            if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(0)
            val_loaders.append(loader)

        if self.rank == 0:
            print("Building TPU list of dataloaders for datasets: ", config["datasets"], flush=True)
        return val_loaders

    def train_dataloader(self) -> DataLoader[Any]:
        if self.prefork_train_dataset is None:
            if not self.train_config.get("datasets") or not all(isinstance(d, str) for d in self.train_config["datasets"]):
                raise ValueError("All train datasets must be provided as a list of strings.")
            dataset = " + ".join(self.train_config["datasets"])
        else:
            dataset = self.prefork_train_dataset
        if self.rank == 0:
            print("Building TPU train dataloader for dataset: ", dataset, flush=True)
        train_loader = get_data_loader(
            dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            drop_last=True,
            fixed_length=self.train_config.get("fixed_length", False),
            accum_steps=self.train_config.get("accum_steps", self.accum_steps),
            debug_enumerate_batches=self.train_config.get("debug_enumerate_batches", False),
            **self._loader_kwargs(self.train_config),
        )
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(0)
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(0)
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(0)
        return train_loader

    def val_dataloader(self):
        return self._dataloader_from_cfg(self.validation_config)

    def test_dataloader(self):
        return self._dataloader_from_cfg(self.test_config)

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict
