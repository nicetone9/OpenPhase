import torch
from torch.nn import functional as F
from torch import sin, cos, atan2, acos
import os
import numpy as np
import math
import random

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = cosine_beta_schedule_discrete(timesteps)
        #elif noise_schedule == 'custom':
            #betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.betas.device != t_int.device:
            self.betas = self.betas.to(t_int.device)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        if self.alphas_bar.device != t_int.device:
            self.alphas_bar = self.alphas_bar.to(t_int.device)
        return self.alphas_bar[t_int.long()]


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def seq_recovery(data, pred_seq):
    '''
    data.x is nature sequence

    '''
    ind = (data.x.argmax(dim=1) == pred_seq.argmax(dim=1))
    recovery = ind.sum() / ind.shape[0]
    return recovery, ind.cpu()


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def set_seed(seed=1024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cal_stats_metric(metric_list):
    """Compute mean and median of a list of metric values."""
    mean_metric = np.mean(metric_list)
    median_metric = np.median(metric_list)
    return mean_metric, median_metric


def enable_dropout(model):
    """Enable dropout during inference (e.g., for MC Dropout)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def get_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of log-probabilities.

    Args:
        log_probs: Tensor of shape (B, V), where V = vocabulary size.

    Returns:
        Tensor of shape (B,) with entropy values.
    """
    probs = torch.exp(log_probs)  # (B, V)
    p_log_p = log_probs * probs   # (B, V)
    entropy = -1 * p_log_p.sum(dim=-1)  # (B,)
    return entropy


def fuse_logits_by_log_probs(log_prob_list, logits_list, temp=1.0):
    """
    Fuse multiple logits using entropy-based weighting.

    Args:
        log_prob_list: list of log-prob tensors, each of shape (B, V)
        logits_list: list of corresponding logits, each of shape (B, V)
        temp: temperature parameter for softmax over entropies

    Returns:
        Fused logits of shape (B, V)
    """
    entropy_list = [get_entropy(log_probs) for log_probs in log_prob_list]  # List of (B,)
    entropy_stack = torch.stack(entropy_list, dim=0)  # (N, B)
    weights = F.softmax(-1 * entropy_stack / temp, dim=0)  # (N, B)

    logits_stack = torch.stack(logits_list, dim=0)  # (N, B, V)
    fused_logits = (weights.unsqueeze(-1) * logits_stack).sum(dim=0)  # (B, V)

    return fused_logits


def sin_mask_ratio_adapter(beta_t_bar, max_deviation=0.2, center=0.5):
    """
    Compute sinusoidal schedule for masking ratio.

    Args:
        beta_t_bar: tensor of normalized time steps in [0,1], shape (B, 1)
        max_deviation: how much the mask ratio deviates from center
        center: central mask ratio

    Returns:
        mask_ratio: tensor of shape (B,) in [center - max_dev, center + max_dev]
    """
    adjusted = beta_t_bar * torch.pi * 0.5           # (B, 1)
    sine = torch.sin(adjusted)                       # (B, 1)
    mask_ratio = center + sine * max_deviation       # (B, 1)
    return mask_ratio.squeeze(1)        