import torch


def gumbel_noise(
        shape: torch.Size,
        device: torch.device,
        eps: float = 1e-9
) -> torch.Tensor:
    """
    生成Gumbel噪声，用于Gumbel-Softmax技巧。
    Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
    """
    uniform_samples = torch.rand(shape, device=device)
    return -torch.log(-torch.log(uniform_samples + eps) + eps)


def linear_anneal(
        current_step: int,
        start_value: float,
        end_value: float,
        total_annealing_steps: int,  # 总共用于退火的步数
        warmup_steps: int = 0  # 退火开始前的预热步数
) -> float:
    """
    对一个值进行线性退火。
    """
    if total_annealing_steps <= 0:  # 避免除以零
        return end_value
    if current_step < warmup_steps:
        return start_value

    # 从预热结束后开始计算退火进度
    effective_step = current_step - warmup_steps
    effective_total_steps = total_annealing_steps - warmup_steps

    if effective_total_steps <= 0:
        return end_value

    progress_ratio = min(effective_step / effective_total_steps, 1.0)
    return start_value + progress_ratio * (end_value - start_value)

