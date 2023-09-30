import numpy as np
import scipy
from scipy.special import erfinv
import ghalton


class Gaussian:
    """Implements a Gaussian pdf. Focus is on efficient multiplication,
       division and sampling."""

    def __init__(self, m=None, P=None, U=None, S=None, Pm=None, L=None):
        """
        Initializes a Gaussian pdf given a valid combination of its parameters.
        Valid combinations are: m-P, m-U, m-S, Pm-P, Pm-U, Pm-S.

        Parameters
        ----------
        m: numpy.array
            Mean
        P: numpy.array
             Precision
        U: numpy.array
             Upper triangular precision factor (U'U = P)
        S: numpy.array
             Covariance matrix
        C: numpy.array
             Upper or lower triangular covariance factor (S = C'C)
        Pm: numpy.array
             Precision times mean such that P*m = Pm
        L: numpy.array
            Lower triangular covariance factor given as 1D array (LL' = S)
        """
        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size
            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif L is not None:
                L = np.asarray(L)
                Lm = np.diag(L[0:self.ndim])
                if 1 < self.ndim < L.shape[0]:  # if full covariance
                    tril_ids = np.tril_indices(self.ndim, -1)
                    Lm[tril_ids[0], tril_ids[1]] = L[self.ndim:]
                self.C = Lm.T
                self.S = np.dot(self.C.T, self.C)
                self.P = np.linalg.inv(self.S)
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))
            else:
                raise ValueError('Precision information missing.')
        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size
            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')
        else:
            raise ValueError('Mean information missing.')

    @property
    def mean(self):
        return self.m

    @mean.setter
    def mean(self, value):
        self.m = value

    @property
    def diag_std(self):
        return np.sqrt(np.diagonal(self.S))

    def __str__(self):
        """Makes a verbose string representation for debug printing."""
        res = 'Gaussian: mean: ' + str(self.m) + \
              '\nDiagonal std: ' + str(self.diag_std)
        return res

    def gen(self, rng, n_samples=1, method='random'):
        """Returns independent samples from the Gaussian."""
        if method == 'random':
            z = rng.standard_normal((n_samples, self.ndim))
            samples = np.dot(z, self.C) + self.m
        elif method == 'halton':
            perms = ghalton.EA_PERMS[:self.ndim]
            sequencer = ghalton.GeneralizedHalton(perms)
            samples = np.array(sequencer.get(int(n_samples) + 1))[1:]
            z = erfinv(2*samples - 1) * np.sqrt(2)
            samples = np.dot(z, self.C) + self.m
        else:
            raise ValueError('Unknown gen method ' + method)
        return samples

    def eval(self, x, ii=None, log=True, debug=False):
        """
        Evaluates the Gaussian pdf.

        Parameters
        ----------
        x: numpy.array
            input data (rows are inputs to evaluate at)
        ii: list
            A list of indices specifying which marginal to evaluate;
            if None, the joint pdf is evaluated
        log: bool
            if True, the log pdf is evaluated

        Returns
        ----------
        res: float
          PDF or log PDF
        """
        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5
        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            eps = 1.e-5 * S.mean() * np.diag(np.random.rand(S.shape[0]))
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S + eps)
            lp = np.array([lp]) if x.shape[0] == 1 else lp
        res = lp if log else np.exp(lp)
        return res

    def __mul__(self, other):
        """Multiply with another Gaussian."""
        assert isinstance(other, Gaussian)
        P = self.P + other.P
        Pm = self.Pm + other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another Gaussian."""
        assert isinstance(other, Gaussian)
        res = self * other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __div__(self, other):
        """Divide by another Gaussian.
           The resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)
        P = self.P - other.P
        Pm = self.Pm - other.Pm
        return Gaussian(P=P, Pm=Pm)

    def __idiv__(self, other):
        """Incrementally divide by another Gaussian.
           The resulting Gaussian might be improper."""
        assert isinstance(other, Gaussian)
        res = self / other
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def __pow__(self, power, modulo=None):
        """Raise Gaussian to a power and get another Gaussian."""
        P = power * self.P
        Pm = power * self.Pm
        return Gaussian(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise Gaussian to a power."""
        res = self**power
        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP
        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another Gaussian,
           i.e. KL(this | other)."""
        assert isinstance(other, Gaussian)
        assert self.ndim == other.ndim
        t1 = np.sum(other.P * self.S)
        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))
        t3 = self.logdetP - other.logdetP
        t = 0.5 * (t1 + t2 + t3 - self.ndim)
        return t
