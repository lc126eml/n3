import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class LinearWarmupCosineAnnealingLR_cut(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_lr=0.0,
        accum_iter=1,
        last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.accum_iter = accum_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Only update every accum_iter steps
        if (self.last_epoch % self.accum_iter) != 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        
        step = self.last_epoch // self.accum_iter  # Effective step after accumulation
        
        if step < self.warmup_steps:
            # Linear warmup
            factor = step / max(1, self.warmup_steps)
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            t = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lrs = [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
                for base_lr in self.base_lrs
            ]
        
        # Apply lr_scale per parameter group
        return [
            lr * group.get("lr_scale", 1.0)
            for lr, group in zip(lrs, self.optimizer.param_groups)
        ]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        super().step()

# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR_grok(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t))
            t = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(t * torch.pi)))
                for base_lr in self.base_lrs
            ]
        
# import math
# import torch
# from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR_chat(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1, eta_min=0):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Number of epochs for linear warmup.
            max_epochs (int): Total number of epochs for the schedule.
            last_epoch (int, optional): The index of last epoch. Default: -1.
            eta_min (float, optional): The minimum learning rate. Default: 0.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR_chat, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase: increase lr linearly
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase: adjust lr with cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
