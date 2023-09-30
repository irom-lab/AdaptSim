import numpy as np
import math
import torch


def sample_uniform(
    rng, range, size=None, ignore_multiple=False, flag_list=True
):
    """Assume [low0, high0, low1, high1,...]. Only consider [low0, high0] if ignore_multiple is True."""
    if ignore_multiple:
        range = range[:2]
    num_range = int(len(range) / 2)
    if num_range > 1:
        range_ind = rng.choice(num_range)
        range = range[range_ind * 2:(range_ind+1) * 2]
    out = rng.uniform(range[0], range[1], size=size)
    if flag_list and isinstance(out, np.ndarray):
        out = out.tolist()
    return out


def sample_integers(rng, range, size=None):
    return rng.integers(range[0], range[1], size=size, endpoint=True)


def normalize(data, lb, ub):
    """To [0,1]"""
    return (data-lb) / (ub-lb)


def unnormalize(data, lb, ub):
    """From [0,1]"""
    return (ub-lb) * data + lb


def unnormalize_tanh(data, lb, ub):
    """From [-1,1]"""
    return (ub-lb) * (data/2 + 0.5) + lb


def wrap_angle(value, low, high):
    assert low < high
    width = high - low
    return value - width * math.floor((value-low) / width)


def standardize(data, eps=1e-8):
    """
    Standardize the input data to make it $~ N(0, 1)$.

    :param data: input ndarray or Tensor
    :param eps: factor for numerical stability
    :return: standardized ndarray or Tensor
    """
    if isinstance(data, np.ndarray):
        return (data - np.mean(data)) / (np.std(data) + float(eps))
    elif isinstance(data, torch.Tensor):
        return (data - torch.mean(data)) / (torch.std(data) + float(eps))
