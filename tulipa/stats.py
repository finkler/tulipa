import numpy as np

from scipy.optimize import curve_fit
from scipy.special import erf, erfinv


def rmsd(yp, y):
    return np.sqrt(np.power(yp - y, 2).sum() / y.size)


class rv_continous:
    def __init__(self, xdata, ydata):
        ydata = np.cumsum(ydata)
        self._pest, pcov = curve_fit(self._cdf, xdata, ydata)
        self.deverr = rmsd(self._cdf(xdata, *self.pest), ydata)

    def cdf(self, x):
        return self._cdf(x, *self._pest)

    def pdf(self, x):
        return self._pdf(x, *self._pest)

    def ppf(self, x):
        return self._ppf(x, *self._pest)


class loglap_gen(rv_continous):
    "Skew-LogLaplace distribution"

    def _cdf(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros(x.size)
        mask = np.logical_and(x >= 0, x < d)
        y[mask] = a / (a + b) * np.power(x[mask] / d, b)
        mask = x >= d
        y[mask] = 1. - b / (a + b) * np.power(d / x[mask], a)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _jac(self, x, d, a, b):
        pass

    def _pdf(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros(x.size)
        c = (a * b) / (d * (a + b))
        mask = np.logical_and(x > 0, x < d)
        y[mask] = c * np.power(x[mask] / d, b - 1.)
        mask = x >= d
        y[mask] = c * np.power(d / x[mask], a + 1.)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _ppf(self, p, d, a, b):
        p = np.asarray(p)
        scalar_input = False
        if p.ndim == 0:
            p = p[None]
            scalar_input = True

        y = np.zeros(p.size)
        ab = a / (a + b)
        mask = np.logical_and(p > 0, p < ab)
        y[mask] = d * np.power((a + b) * p[mask] / a, 1. / b)
        mask = np.logical_and(ab < p, p < 1)
        y[mask] = d * np.power((a + b) * (1. - p[mask]) / b, -1. / a)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _stats(self, d, a, b, moments='mv'):
        mu, mu2, g1, g2 = None, None, None, None
        if a > 1:
            mu = d * (a * b) / ((a - 1.) * (b + 1.))
        if a > 2:
            mu2 = d**2 * ((a * b) / ((a - 2.) * (b + 2.)) -
                    ((a * b) / ((a - 1.) * (b + 1.)))**2)
        return mu, mu2, g1, g2


class lognorm_gen(rv_continous):
    "Log-normal distribution"

    def _cdf(self, x, m, s):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = .5 + .5 * erf((np.log(x) - m) / (np.sqrt(2.) * s))

        if scalar_input:
            return np.squeeze(y)
        return y

    def _jac(self, x, m, s):
        pass

    def _pdf(self, x, m, s):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = 1. / (x * s * np.sqrt(2. * np.pi)) * np.exp(-np.power(np.log(x) - m, 2) / (2. * np.power(s, 2)))

        if scalar_input:
            return np.squeeze(y)
        return y

    def _ppf(self, p, m, s):
        p = np.asarray(p)
        scalar_input = False
        if p.ndim == 0:
            p = p[None]
            scalar_input = True

        y = m + s * np.sqrt(2.) * erfinv(2. * p - 1.)

        if scalar_input:
            return np.squeeze(y)
        return y

    def _stats(self, m, s, moments='mv'):
        g1, g2 = None, None
        s2 = s**2
        mu = np.exp(m + .5 * s2)
        mu2 = (np.exp(s2) - 1.) *  np.exp(2. * m + s2)
        return mu, mu2, g1, g2
