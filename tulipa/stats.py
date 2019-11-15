import numpy as np

from scipy.stats import rv_continuous


def rmsd(yf, y):
    return np.sqrt(np.power(yf - y, 2).sum() / y.size)


class loglap_gen(rv_continuous):
    "Skew-LogLaplace distribution"
    def _cdf(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros(x.size)
        mask = x < d
        y[mask] = a / (a + b) * (x[mask] / d)**b
        mask ^= True
        y[mask] = 1. - b / (a + b) * (d / x[mask])**a

        if scalar_input:
            return np.squeeze(y)
        return y

    def _pdf(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros(x.size)
        c = (a * b) / (d * (a + b))
        mask = x < d
        y[mask] = c * (x[mask] / d)**(b - 1.)
        mask ^= True
        y[mask] = c * (d / x[mask])**(a + 1.)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _ppf(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros(x.size)
        mask = x < a / (a + b)
        y[mask] = d * ((a + b) * x[mask] / a)**(1. / b)
        mask ^= True
        y[mask] = d * ((a + b) * (1. - x[mask]) / b)**(-1. / a)

        if scalar_input:
            return np.squeeze(y)
        return y


loglap = loglap_gen(a=0, name="loglap")
