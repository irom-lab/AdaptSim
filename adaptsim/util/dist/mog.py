import numpy as np
import scipy
from scipy.special import logsumexp  # was in scipy.misc until 1.1.0

from .gaussian import Gaussian
from .discrete import discrete_sample


class MoG:
    """Implements a mixture of Gaussians."""

    def __init__(
        self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None, Ls=None
    ):
        """
        Creates a mog with a valid combination of parameters.

        Parameters
        ----------
        a: list or numpy.array
            Mixing coefficients (mixture weights).
        ms: numpy.array
            Component means.
        Ps: numpy.array
            Precisions
        Us: numpy.array
            Precision factors (U'U = P)
        Ss: numpy.array
            Covariances
        xs: list or numpy.array
            Gaussian variables
        Ls: numpy.array
            Lower-triangular covariance factor (L*L' = S)
        """
        if ms is not None:
            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in zip(ms, Ps)]
            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in zip(ms, Us)]
            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in zip(ms, Ss)]
            elif Ls is not None:
                self.xs = [Gaussian(m=m, L=L) for m, L in zip(ms, Ls)]
            else:
                raise ValueError('Precision information missing.')
        elif xs is not None:
            self.xs = xs
        else:
            raise ValueError('Mean information missing.')
        self.a = np.asarray(a)
        self.ndim = self.xs[0].ndim
        self.n_components = len(self.xs)
        self.ncomp = self.n_components

    @property
    def weights(self):
        return self.a

    @property
    def components(self):
        return self.xs

    @property
    def mean(self):
        """Use the component with highest weight"""
        component_ind = np.argmax(self.weights)
        return self.xs[component_ind].mean

    def gen(self, rng, n_samples=1, method='random'):
        """Generates independent samples from mog."""
        ii = discrete_sample(rng, self.a, n_samples)
        ns = [np.sum((ii == i).astype(int)) for i in range(self.n_components)]
        samples = [
            x.gen(rng, n_samples=n, method=method)
            for x, n in zip(self.xs, ns)
        ]
        samples = np.concatenate(samples, axis=0)
        return samples

    def eval(self, x, ii=None, log=True, debug=False):
        """
        Evaluates the mog pdf.
        x: rows are inputs to evaluate at
        ii: a list of indices specifying which marginal to evaluate;
                   if None, the joint pdf is evaluated
        log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """
        ps = np.array([
            self.xs[ix].eval(x, ii, log) for ix in range(len(self.a))
        ]).T
        if log:
            res = scipy.special.logsumexp(ps + np.log(self.a), axis=1)
        else:
            res = np.dot(ps, self.a)
        if debug:
            print('weights\n', self.a, '\nps\n', ps, '\nres\n', res)
        return res

    def __str__(self):
        """Makes a verbose string representation for debug printing."""
        mus = np.array([gauss.m.tolist() for gauss in self.xs])
        diagS = np.array([np.diagonal(gauss.S).tolist() for gauss in self.xs])
        res = 'MoG:\nweights:\n' + str(self.a) + '\nmeans:\n' + str(mus) + \
              '\ndiagS:\n' + str(diagS)
        return res

    def __mul__(self, other):
        """Multiplies by a single Gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x * other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m))
            lcs[i] += np.dot(other.m, np.dot(other.P, other.m))
            lcs[i] -= np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """Incrementally multiplies by a single Gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.a = res.a
        self.xs = res.xs
        return res

    def __div__(self, other):
        """Divides by a single Gaussian."""
        assert isinstance(other, Gaussian)
        ys = [x / other for x in self.xs]
        lcs = np.empty_like(self.a)
        for i, (x, y) in enumerate(zip(self.xs, ys)):
            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m))
            lcs[i] -= np.dot(other.m, np.dot(other.P, other.m))
            lcs[i] -= np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5
        la = np.log(self.a) + lcs
        la -= logsumexp(la)
        a = np.exp(la)
        return MoG(a=a, xs=ys)

    def __idiv__(self, other):
        """Incrementally divides by a single Gaussian."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.a = res.a
        self.xs = res.xs
        return res

    def calc_mean_and_cov(self):
        """Calculates the mean vector and the covariance matrix of the MoG."""
        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms)[np.newaxis, :])
        Ss = [x.sigma for x in self.xs]
        S = np.dot(self.a, np.array(Ss)[np.newaxis, :])
        return m, S

    def project_to_gaussian(self):
        """Returns a Gaussian with the same mean and precision as the MoG."""
        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S)

    def prune_negligible_components(self, threshold):
        """Removes all components with mixing coefficients < threshold."""
        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size
        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000):
        """Estimates the kl from this to another pdf,
           i.e. KL(this | other), using Monte Carlo."""
        x = self.gen(n_samples)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq
        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)
        return res, err
