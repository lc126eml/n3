"""CPU checks: PYTHONPATH=training python -m unittest train_utils.test_unfreeze."""

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn

from trainer import Trainer
from train_utils.freeze import freeze_modules, select_unfreeze_modules, validate_unfreeze_configs


def make_model():
    model = nn.Module()
    model.aggregator = nn.Module()
    for name in ("frame_blocks", "inter_frame_blocks"):
        setattr(model.aggregator, name, nn.ModuleList([nn.Linear(2, 2) for _ in range(24)]))
    model.aggregator.patch_embed = nn.Linear(2, 2)
    model.aggregator.register_parameter("native", nn.Parameter(torch.ones(2), requires_grad=False))
    model.dense_head = nn.Module()
    for name in ("point_proj", "point_proj_conf", "cam_point_proj", "shared", "proj"):
        setattr(model.dense_head, name, nn.Linear(2, 2))
    model.camera_head = nn.Linear(2, 2)
    return model


class UnfreezeTests(unittest.TestCase):
    def setUp(self):
        self.optim = OmegaConf.load(
            Path(__file__).resolve().parents[1] / "configs/default_24.yaml"
        ).optim

    def trainer(self, optim=None, compiled=False):
        trainer = Trainer.__new__(Trainer)
        trainer.model = make_model()
        trainer._native_frozen_param_names = {"aggregator.native"}
        trainer.optim_conf = copy.deepcopy(self.optim if optim is None else optim)
        validate_unfreeze_configs(trainer.model, trainer.optim_conf.get("unfreeze_configs", []))
        freeze_modules(trainer.model, ["*"])
        if compiled:
            trainer.model = torch.compile(trainer.model, backend="eager")
        trainer.epoch = 0
        trainer.discount = 1.0
        trainer._apply_train_sampler_batch_cost_discount = lambda value: setattr(trainer, "discount", value)
        trainer.cfg = OmegaConf.create({"loss": {"weight": 0}})
        trainer.loss = SimpleNamespace(weight=0)
        return trainer

    def state(self, trainer):
        return ({name for name, param in trainer.model.named_parameters() if param.requires_grad},
                trainer.discount, trainer.loss.weight)

    def test_schedule_and_optimizer(self):
        trainer = self.trainer()
        model = trainer.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        for epoch, discount in ((0, 1.0), (1, 0.8), (2, 0.9), (3, 0.75), (5, 0.7), (15, 0.5)):
            trainer.epoch = epoch
            trainer.end_warmup()
            self.assertEqual(trainer.discount, discount)
            for name, param in model.named_parameters():
                if name == "aggregator.native":
                    expected = False
                elif name.startswith("dense_head.point_proj"):
                    expected = True
                elif "_head." in name:
                    expected = epoch >= 2
                elif "_blocks." in name:
                    index = int(name.split(".")[2])
                    expected = epoch >= 5 or (epoch >= 3 and index >= 12) or (epoch >= 1 and index >= 16)
                else:
                    expected = epoch >= 5
                self.assertEqual(param.requires_grad, expected, (epoch, name))

        # Gradients traverse frozen downstream layers to newly trainable blocks.
        trainer = self.trainer()
        model = trainer.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        trainer.epoch = 0
        trainer.end_warmup()
        trainer.epoch = 1
        trainer.end_warmup()
        weight = model.aggregator.frame_blocks[16].weight
        before = weight.detach().clone()
        x = model.aggregator.frame_blocks[16](torch.ones(1, 2))
        x = model.dense_head.shared(x)
        model.dense_head.point_proj(x).sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertIsNone(model.dense_head.shared.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, weight))

    def test_resume_and_ordering(self):
        # Reverse input order, overlap stages, and mix attribute transitions.
        optim = copy.deepcopy(self.optim)
        optim.unfreeze_configs = list(reversed(optim.unfreeze_configs))
        optim.warmup_configs = [
            {"epoch": 3, "attr": "loss.weight", "value": 3},
            {"epoch": 1, "attr": "loss.weight", "value": 1},
            {"epoch": 3, "attr": "loss.weight", "value": 4},
        ]
        for resumed_epoch in (0, 1, 2, 3, 4, 5, 6, 15, 16):
            continuous = self.trainer(optim)
            for epoch in range(resumed_epoch):
                continuous.epoch = epoch
                continuous.end_warmup()
            resumed = self.trainer(optim)
            resumed.epoch = resumed_epoch
            resumed.end_warmup(replay=True)
            self.assertEqual(self.state(continuous), self.state(resumed))
            continuous.epoch = resumed_epoch
            continuous.end_warmup()
            resumed.end_warmup()
            self.assertEqual(self.state(continuous), self.state(resumed))
            resumed.end_warmup(replay=True)
            resumed.end_warmup()
            self.assertEqual(self.state(continuous), self.state(resumed))

    def test_equal_epoch_and_legacy(self):
        optim = copy.deepcopy(self.optim)
        optim.warmup_epochs = 0
        optim.unfreeze_configs.append({
            "epoch": 0, "modules": {"patterns": ["camera_head"]}, "batch_cost_discount": 0.6,
        })
        trainer = self.trainer(optim)
        trainer.end_warmup()
        self.assertEqual(trainer.discount, 0.6)
        for empty in (None, []):
            legacy = copy.deepcopy(self.optim)
            if empty is None:
                del legacy.unfreeze_configs
            else:
                legacy.unfreeze_configs = empty
            trainer = self.trainer(legacy)
            trainer.epoch = 15
            trainer.end_warmup()
            self.assertEqual(trainer.discount, 0.5)
            self.assertEqual(len(self.state(trainer)[0]), len(list(trainer.model.parameters())) - 1)

    def test_compiled_native_and_patterns(self):
        trainer = self.trainer(compiled=True)
        trainer.end_warmup()
        model = trainer.model._orig_mod
        self.assertTrue(model.dense_head.point_proj_conf.weight.requires_grad)
        self.assertFalse(model.dense_head.cam_point_proj.weight.requires_grad)
        trainer.epoch = 16
        trainer.end_warmup(replay=True)
        self.assertFalse(model.aggregator.native.requires_grad)
        selected = select_unfreeze_modules(model, {
            "patterns": ["camera_head"], "frame_block_ranges": [[0, 1]],
        })
        self.assertEqual(len(selected), 3)

    def test_invalid_configs(self):
        for changes in (
            {"epoch": -1}, {"epoch": 1.5}, {"batch_cost_discount": 0},
            {"batch_cost_discount": float("nan")}, {"batch_cost_discount": float("inf")},
            {"modules": {"patterns": ["missing"]}}, {"modules": {}},
            {"modules": {"frame_block_ranges": [[0, 25]]}},
            {"modules": {"frame_block_ranges": [[2, 2]]}},
            {"modules": {"frame_block_ranges": [[-1, 2]]}},
            {"modules": {"frame_block_ranges": [[0.5, 2]]}},
        ):
            conf = {"epoch": 0, "modules": {"patterns": ["camera_head"]}, "batch_cost_discount": 1.0}
            conf.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_unfreeze_configs(make_model(), [conf])


if __name__ == "__main__":
    unittest.main()
