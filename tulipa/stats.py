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
        mask = 0 <= x < d
        y[mask] = a / (a + b) * np.power(x[mask] / d, b)
        mask = x >= d
        y[mask] = 1. - b / (a + b) * np.power(d / x[mask], a)

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
        mask = 0 < x < d
        y[mask] = c * np.power(x[mask] / d, b - 1.)
        mask = x >= d
        y[mask] = c * np.power(d / x[mask], a + 1.)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _ppf(self, q, d, a, b):
        q = np.asarray(q)
        scalar_input = False
        if q.ndim == 0:
            q = q[None]
            scalar_input = True

        y = np.zeros(q.size)
        ab = a / (a + b)
        mask = 0 < q < ab
        y[mask] = d * np.power((a + b) * q[mask] / a, 1. / b)
        mask = ab < q < 1
        y[mask] = d * np.power((a + b) * (1. - q[mask]) / b, -1. / a)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _stats(self, d, a, b, moments='mv'):
        stats = np.zeros(len(moments))
        if 'm' in moments and a > 1:
            m = d * (a * b) / ((a - 1.) * (b + 1.))
            stats[moments.index('m')] = m
        if 'v' in moments and a > 2:
            v = d**2 * ((a * b) / ((a - 2.) * (b + 2.)) -
                        ((a * b) / ((a - 1.) * (b + 1.)))**2)
            stats[moments.index('v')] = v
        return stats


loglap = loglap_gen(a=0, name="loglap")
