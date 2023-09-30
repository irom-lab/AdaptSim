import numpy as np
import ghalton


class Uniform:
    """Implements a Uniform pdf.

    Parameters
    ----------
    lb_array: numpy.array
      Lower bounds
    ub_array: numpy.array
      Upper bounds
    """

    def __init__(self, lb_array=None, ub_array=None):
        self.lb_array = lb_array
        self.ub_array = ub_array
        self.cur_index = 0
        self.cached_samples = None
        assert len(lb_array) == len(ub_array)
        self.param_dim = len(lb_array)

    @property
    def mean(self):
        return (self.lb + self.ub) / 2

    @property
    def lb(self):
        return self.lb_array

    @property
    def ub(self):
        return self.ub_array

    def __str__(self):
        """Makes a verbose string representation for debug printing."""
        res = 'Uniform: lower bounds: ' + str(self.lb_array) + \
              '\nupper bounds: ' + str(self.ub_array)
        return res

    def generate_halton_samples(self, n_samples=1000):
        """Generates Halton samples.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate

        Returns
        ----------
        h_sample: numpy.array
          A vector of samples
        """
        domain = np.zeros((2, len(self.ub_array)))
        for ix in range(self.param_dim):
            domain[0][ix] = self.lb_array[0]
            domain[1][ix] = self.ub_array[1]
        dim = domain.shape[1]
        perms = ghalton.EA_PERMS[:dim]
        sequencer = ghalton.GeneralizedHalton(perms)
        h_sample = np.array(sequencer.get(n_samples + 1))[1:]
        if dim == 1:
            h_sample = domain[0] + h_sample * (domain[1] - domain[0])
        else:
            h_sample = domain[0, :] + h_sample * (domain[1, :] - domain[0, :])
        return h_sample

    def gen(self, rng, n_samples=1, method='random'):
        """Generates samples.

        Parameters
        ----------
        n_samples: int, optional
            Number of samples to generate
        method: string, optional; 'random' or 'halton'
            Use Halton sampling if 'halton', random uniform if 'random'.

        Returns
        ----------
        result: numpy.array
          A vector of samples
        """
        result = None
        if method == 'halton':
            result = self.generate_halton_samples(n_samples=n_samples)
        elif method == 'random':
            for ix in range(len(self.lb_array)):
                samples = rng.uniform(
                    self.lb_array[ix], self.ub_array[ix], size=n_samples
                )
                if result is None:
                    result = samples
                else:
                    result = np.concatenate((result, samples), axis=0)
        else:
            raise ValueError('Unknown gen method ' + method)
        return result.reshape(-1, len(self.lb_array))

    def eval(self, x, ii=None, log=True, debug=False):
        """Evaluates Uniform PDF

        Parameters
        ----------
        x : int or list or np.array
            Rows are inputs to evaluate at
        ii : list
            A list of indices specifying which marginal to evaluate.
            If None, the joint pdf is evaluated
        log : bool, defaulting to True
            If True, the log pdf is evaluated

        Returns
        -------
        p: float
          PDF or log PDF
        """
        if ii is None:
            ii = np.arange(self.param_dim)
        N = np.atleast_2d(x).shape[0]
        p = 1 / np.prod(self.ub_array[ii] - self.lb_array[ii])
        p = p * np.ones((N,))  # broadcasting
        # truncation of density
        ind = (x > self.lb_array[ii]) & (x < self.ub_array[ii])
        p[np.prod(ind, axis=1) == 0] = 0
        if log:
            if not ind.any():
                raise ValueError('log prob. not defined outside of truncation')
            else:
                return np.log(p)
        else:
            return p
