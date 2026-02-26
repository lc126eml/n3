import sys
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader

# Assuming these are in the python path
from training.datasets import get_data_loader
from training.datasets import *
from training.datasets.utils.transforms import SeqColorJitter


class StandaloneMultiViewDataModule:
    """
    A framework-agnostic data module for creating multiview dataloaders.

    This class mirrors the logic of the original MultiViewDUSt3RDataModule
    but does not depend on PyTorch Lightning. It instantiates datasets
    from string configurations provided via a config file.
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
        **kwargs,
    ) -> None:
        """Initialize a StandaloneMultiViewDataModule."""
        self.train_config = train_config
        self.validation_config = validation_config
        self.test_config = test_config
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.seed = seed
        self.accum_steps = int(kwargs.get("accum_steps", 1))

    def _dataloader_from_cfg(self, config: dict):
        """Creates a combined dataloader from a list of dataset strings."""
        if not config.get("datasets") or not all(isinstance(d, str) for d in config["datasets"]):
            raise ValueError("All datasets must be provided as a list of strings.")

        # WARNING: Using eval is a security risk. This is preserved from the
        # original implementation as requested.
        try:
            val_datasets = [eval(dataset) for dataset in config["datasets"]]
        except NameError as e:
            print(f"Error during eval: {e}. Make sure all dataset classes (e.g., HyperSim_Multi) and transforms (e.g., SeqColorJitter) are imported.")
            raise

        val_loaders = []
        for dataset in val_datasets:
            val_loaders.append(
                get_data_loader(
                    dataset,
                    batch_size=config["batch_size"],
                    num_workers=config["num_workers"],
                    pin_mem=self.pin_memory,
                    shuffle=False,
                    drop_last=False,
                    seed=self.seed,
                    persistent_workers=self.persistent_workers,
                    prefetch_factor=self.prefetch_factor,
                )
            )

        for loader in val_loaders:
            if hasattr(loader.dataset, "set_epoch"):
                print(f"Dataset: {loader.dataset} | Length: {len(loader.dataset)}")
                loader.dataset.set_epoch(0)
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

        print("Building list of dataloaders for datasets: ", config["datasets"])
        return val_loaders

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        if not self.train_config.get("datasets") or not all(isinstance(d, str) for d in self.train_config["datasets"]):
            raise ValueError("All train datasets must be provided as a list of strings.")

        train_datasets_concat = " + ".join(self.train_config["datasets"])
        print("Building train Data loader for dataset: ", train_datasets_concat)
        train_loader = get_data_loader(
            train_datasets_concat,
            batch_size=self.train_config["batch_size"],
            num_workers=self.train_config["num_workers"],
            pin_mem=self.pin_memory,
            shuffle=True,
            drop_last=True,
            fixed_length=self.train_config.get("fixed_length", False),
            seed=self.seed,
            accum_steps=self.train_config.get("accum_steps", self.accum_steps),
            debug_enumerate_batches=self.train_config.get("debug_enumerate_batches", False),
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return self._dataloader_from_cfg(self.validation_config)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return self._dataloader_from_cfg(self.test_config)
