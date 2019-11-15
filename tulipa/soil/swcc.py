import numpy as np


class swcc_model:
    def Theta(self, h):
        return self._Theta(h, *self.params)


class vgmodel_gen(swcc_model):
    def __init__(self):
        self.params = [0.005, 2.0]
        self.bounds = ([0, 1], np.inf)

    def _jac(self, h, a, n):
        h = np.asarray(h)
        scalar_input = False
        if h.ndim == 0:
            h = h[None]
            scalar_input = True

        j = np.zeros((h.size, 2))
        m = 1. - 1. / n
        xi = 1. / (1. + np.power(a * h, n))
        j[:, 0] = -h * np.power(a * h, n - 1.) * np.power(xi,
                                                          2. - 1. / n) * m * n
        j[:, 1] = np.power(xi, m) * (
            -np.power(a * h, n) * m * np.log(a * h) /
            (1. + np.power(a * h, n)) + np.log(xi) / n**2)

        if scalar_input:
            return np.squeeze(j)
        return j

    def _Theta(self, h, a, n):
        m = 1. - 1. / n
        return (1. / (1. + (a * h)**n))**m


class fxmodel_gen(swcc_model):
    def __init__(self):
        self.params = [0.005, 0.5, 2.0]
        self.bounds = ([0, 0, 1], np.inf)

    def _jac(self, h, a, m, n):
        h = np.asarray(h)
        scalar_input = False
        if h.ndim == 0:
            h = h[None]
            scalar_input = True

        j = np.zeros((h.size, 3))
        c1 = np.exp(1.) + np.power(a * h, n)
        l1 = np.log(c1)
        j[:, 0] = -h * np.power(a * h, n - 1.) * m * n * np.power(l1,
                                                                  -m - 1.) / c1
        j[:, 1] = -np.power(l1, -m) * np.log(l1)
        j[:, 2] = -np.power(a * h, n) * m * np.log(a * h) * np.power(
            l1, -m - 1.) / c1

        if scalar_input:
            return np.squeeze(j)
        return j

    def _Theta(self, h, a, m, n):
        return np.power(np.log(np.exp(1.) + np.power(a * h, n)), -m)


fxmodel = fxmodel_gen()
vgmodel = vgmodel_gen()
