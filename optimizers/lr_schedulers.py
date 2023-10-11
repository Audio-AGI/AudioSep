from functools import partial
from typing import Callable


def linear_warm_up(
    step: int, 
    warm_up_steps: int, 
    reduce_lr_steps: int
) -> float:
    r"""Get linear warm up scheduler for LambdaLR.

    Args:
        step (int): global step
        warm_up_steps (int): steps for warm up
        reduce_lr_steps (int): reduce learning rate by a factor of 0.9 #reduce_lr_steps step

    .. code-block: python
        >>> lr_lambda = partial(linear_warm_up, warm_up_steps=1000, reduce_lr_steps=10000)
        >>> from torch.optim.lr_scheduler import LambdaLR
        >>> LambdaLR(optimizer, lr_lambda)

    Returns:
        lr_scale (float): learning rate scaler
    """

    if step <= warm_up_steps:
        lr_scale = step / warm_up_steps
    else:
        lr_scale = 0.9 ** (step // reduce_lr_steps)

    return lr_scale


def constant_warm_up(
    step: int, 
    warm_up_steps: int, 
    reduce_lr_steps: int
) -> float:
    r"""Get constant warm up scheduler for LambdaLR.

    Args:
        step (int): global step
        warm_up_steps (int): steps for warm up
        reduce_lr_steps (int): reduce learning rate by a factor of 0.9 #reduce_lr_steps step

    .. code-block: python
        >>> lr_lambda = partial(constant_warm_up, warm_up_steps=1000, reduce_lr_steps=10000)
        >>> from torch.optim.lr_scheduler import LambdaLR
        >>> LambdaLR(optimizer, lr_lambda)

    Returns:
        lr_scale (float): learning rate scaler
    """
    
    if 0 <= step < warm_up_steps:
        lr_scale = 0.001

    elif warm_up_steps <= step < 2 * warm_up_steps:
        lr_scale = 0.01

    elif 2 * warm_up_steps <= step < 3 * warm_up_steps:
        lr_scale = 0.1

    else:
        lr_scale = 1

    return lr_scale


def get_lr_lambda(
    lr_lambda_type: str, 
    **kwargs
) -> Callable:
    r"""Get learning scheduler.

    Args:
        lr_lambda_type (str), e.g., "constant_warm_up" | "linear_warm_up"

    Returns:
        lr_lambda_func (Callable)
    """
    if lr_lambda_type == "constant_warm_up":

        lr_lambda_func = partial(
            constant_warm_up, 
            warm_up_steps=kwargs["warm_up_steps"], 
            reduce_lr_steps=kwargs["reduce_lr_steps"],
        )

    elif lr_lambda_type == "linear_warm_up":

        lr_lambda_func = partial(
            linear_warm_up, 
            warm_up_steps=kwargs["warm_up_steps"], 
            reduce_lr_steps=kwargs["reduce_lr_steps"],
        )

    else:
        raise NotImplementedError

    return lr_lambda_func
