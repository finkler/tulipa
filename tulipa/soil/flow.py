import numpy as np


class hc_model:
    def __init__(self, n, d, T=15):
        self._d = d
        self.g = 980.0
        self.mu = self._mu(T)
        self.n = n
        self.rho = self._rho(T)

    def _mu(self, T):
        return -7.0e-8 * T**3 + 1.002e-5 * T**2 - 5.7e-4 * T + 0.0178

    def _rho(self, T):
        return 3.1e-8 * T**3 - 7.0e-6 * T**2 + 4.19e-5 * T + 0.99985

    def d(self, x):
        return self._d(x) * 0.1

    def K(self):
        return (self.rho * self.g / self.mu) * self.N * self._phi() * self.de


class hazen(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 10.0 * self.mu / (self.rho * self.g)
        self.de = self.d(0.1)

    def _phi(self):
        return 1.0


def slichter(hc_model):
    def __init__(self, n, d):
        super().__init__(n, d)
        self.N = 1e-2
        self.de = self.d(0.1)

    def _phi(self):
        return np.power(self.n, 3.287)
