import torch
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupLR(LambdaLR):
    """LR Scheduling function which is increase lr on warmup steps and decrease on normal steps"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        reduce_rate: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            optimizer: torch optimizer
            num_warmup_steps: number of warmup steps
            num_training_steps: number of whole training steps
            reduce_rate: min Learning Rate / max Learning Rate
        """
        self.num_warmup_steps = num_warmup_steps
        self.decrement = (1.0 - reduce_rate) / (num_training_steps - num_warmup_steps)
        self.reduce_rate = reduce_rate

        super().__init__(optimizer, self._get_lr, last_epoch=last_epoch, verbose=verbose)

    def _get_lr(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return current_step / self.num_warmup_steps
        return max(1.0 - self.decrement * (current_step - self.num_warmup_steps), self.reduce_rate)
