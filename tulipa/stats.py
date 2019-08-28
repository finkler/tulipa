import numpy as np

from scipy.optimize import curve_fit
from scipy.special import erf, erfinv


class rv_distribution:
    def cdf(self, x):
        return self._cdf(x, *self.shapes)

    def pdf(self, x):
        return self._pdf(x, *self.shapes)

    def ppf(self, x):
        return self._ppf(x, *self.shapes)


class loglap_gen(rv_distribution):
    def __init__(self, d=1, a=1, b=0):
        self.shapes = np.array([d, a, b])
        self.bounds = ([0, 0, 0], np.inf)

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

    def _fit(self, x, d, a, b):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros((x.size, 3))
        mask = x < d
        c = (x[mask] / d)**b
        y[mask, 0] = -a * b * c / (a * d + b * d)
        y[mask, 1] = b * c / (a + b)**2
        y[mask, 2] = a * c * (-1. + (a + b) * np.log(x[mask] / d)) / (a + b)**2
        mask ^= True
        c = (d / x[mask])**a
        y[mask, 0] = -a * b * c / (a * d + b * d)
        y[mask, 1] = -b * c * (-1. +
                               (a + b) * np.log(d / x[mask])) / (a + b)**2
        y[mask, 2] = -a * c / (a + b)**2

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


class lognorm_gen(rv_distribution):
    def __init__(self, s=1, m=0):
        self.shapes = np.array([s, m])
        self.bounds = ([0, -np.inf], np.inf)

    def _cdf(self, x, s, m):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = .5 + .5 * erf((np.log(x) - m) / (s * np.sqrt(2.)))

        if scalar_input:
            return np.squeeze(y)
        return y

    def _fit(self, x, s, m):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.zeros((x.size, 2))
        c = -0.398942 * np.exp(-(m - np.log(x))**2 / (2. * s**2))
        y[:, 0] = c * (np.log(x) - m) / s**2
        y[:, 1] = c / s

        if scalar_input:
            return np.squeeze(y)
        return y

    def _pdf(self, x, s, m):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = 1. / (
            x * s * np.sqrt(2. * np.pi)) * np.exp(-(np.log(x) - m)**2 /
                                                  (2. * s**2))

        if scalar_input:
            return np.squeeze(y)
        return y

    def _ppf(self, x, s, m):
        x = np.asarray(x)
        scalar_input = False
        if x.ndim == 0:
            x = x[None]
            scalar_input = True

        y = np.exp(m + 1.41421 * s * erfinv(-1. + 2. * x))

        if scalar_input:
            return np.squeeze(y)
        return y


def generic_fit(data):
    # data = data[data[:, 0].argsort()]
    xdata = data[:, 0]
    ydata = np.cumsum(data[:, 1])
    rv_continous = [loglap_gen(), lognorm_gen()]
    perr = []
    for rvc in rv_continous:
        rvc.shapes, pcov = curve_fit(
            rvc._cdf,
            xdata,
            ydata,
            bounds=rvc.bounds,
            p0=rvc.shapes,
            jac=rvc._fit)
        perr.append(np.sqrt(np.abs(np.diag(pcov)).mean()))
    return rv_continous[np.argmin(perr)]
