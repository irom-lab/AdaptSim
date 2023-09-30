import numpy as np


def discrete_sample(rng, p, n_samples=1):
    """Samples from a discrete distribution.

    Parameters
    ----------
    p: a distribution with N elements
    n_samples: number of samples

    Returns
    ----------
    res: vector of samples
    """
    cumul_distr = np.cumsum(p[:-1])[np.newaxis, :]
    rnd = rng.random((n_samples, 1))
    res = np.sum((rnd > cumul_distr).astype(int), axis=1)
    return res
