import numpy as np

from tulipa.soil.sediment import phi


class hc_model:
    def __init__(self, n, d, unit='mm', T=15.0):
        factor = {'cm': 1.0, 'mm': 0.1, 'um': 0.0001}
        self._d = d
        self._factor = factor[unit]
        self.g = 980.0
        self.mu = self._mu(T)
        self.n = n
        self.rho = self._rho(T)
        self.tau = self._tau(T)

    def _mu(self, T):
        return -7.0e-8 * T**3 + 1.002e-5 * T**2 - 5.7e-4 * T + 0.0178

    def _rho(self, T):
        return 3.1e-8 * T**3 - 7.0e-6 * T**2 + 4.19e-5 * T + 0.99985

    def _tau(self, T):
        return 1.093e-4 * T**2 + 2.102e-2 * T + 0.5889

    def d(self, x):
        return self._d(x) * self._factor

    def K(self):
        return (self.rho * self.g / self.mu) * self.k()

    def k(self):
        return self.N * self.phi * self.de


class barr(hc_model):
    def __init__(self, n, d, grains='spherical'):
        super().__init__(n, d)
        c = 1.0 if grains == 'spherical' else 1.35
        self.N = 1.0 / (180.0 * np.power(c, 2))
        self.de = self.d(0.1)
        self.phi = np.power(self.n, 3) / np.power(1.0 - n, 2)


class beyer(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.de = self.d(0.1)
        U = self.d(0.6) / self.de
        self.N = 5.2e-4 * np.log(500.0 / U)
        self.phi = 1.0


class chapuis(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        xi = self.n / (1.0 - self.n)
        self.N = self.mu / (self.g * self.rho)
        self.de = np.power(
            self.d(0.1), 0.5 * np.power(10.0, 0.5504 - 0.2637 * xi))
        self.phi = np.power(10.0, 1.291 * xi - 0.6435)


class hazen(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 10.0 * self.mu / (self.rho * self.g)
        self.de = self.d(0.1)
        self.phi = 1.0


class krumbein(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 7.501e-6
        d5 = phi(self.d(0.05) * 10.0)
        d16 = phi(self.d(0.16) * 10.0)
        d50 = phi(self.d(0.5) * 10.0)
        d84 = phi(self.d(0.84) * 10.0)
        d95 = phi(self.d(0.95) * 10.0)
        si = (d84 - d16) / 4.0 + (d95 - d5) / 6.6
        self.de = np.exp(-1.31 * si)
        self.phi = np.power(2.0, (d16 + d50 + d84) / 3.0)


class sauerbrei(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 3.75e-5 * self.tau
        self.de = self.d(0.1)
        self.phi = np.power(self.n, 3) / np.power(1.0 - n, 2)


class slichter(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 1e-2
        self.de = self.d(0.1)
        self.phi = np.power(self.n, 3.287)


class terzaghi(hc_model):
    def __init__(self, n, d, grains='smooth'):
        super().__init__(n, d)
        self.N = 10.7e-3 if grains == 'smooth' else 6.1e-3
        self.de = self.d(0.1)
        self.phi = np.power((self.n - 0.13) / np.cbrt(1.0 - self.n), 2)


class usbr(hc_model):
    def __init__(self, n, d, grains='smooth'):
        super().__init__(n, d)
        self.N = 4.8e-4 * np.power(10.0, 0.3)
        self.de = np.power(self.d(0.2), 1.15)
        self.phi = 1.0


def estimate(model, n, d):
    for cls in hc_model.__subclasses__():
        if model == cls.__name__:
            return cls(n, d).K()
    return None
