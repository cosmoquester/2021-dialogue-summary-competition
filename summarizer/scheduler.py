from typing import Callable


def get_linear_schedule_with_warmup(num_warmup_steps: int, num_training_steps: int, reduce_rate: float) -> Callable:
    """LR Scheduling function which is increase lr on warmup steps and decrease on normal steps

    Args:
        num_warmup_steps: number of warmup steps
        num_training_steps: number of whole training steps
        reduce_rate: min Learning Rate / max Learning Rate 비율
    Returns:
        Scheduling function (use with LambdaLR)
    """
    decrement = (1.0 - reduce_rate) / (num_training_steps - num_warmup_steps)

    def _lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps
        return max(1.0 - decrement * (current_step - num_warmup_steps), reduce_rate)

    return _lr_lambda
