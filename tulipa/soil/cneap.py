import numpy as np

from scipy.optimize import bisect, curve_fit
from tulipa.soil.swcc import fxmodel, vgmodel

alpha = 1.38
Cc = 130.0  # um-kPa


# kPa to cm
def pressurehead(psi):
    g = 9.81
    rho = 997.0
    return (psi * 1e3 / (g * rho)) * 100.0


class CNEAP:
    def __init__(self, gsd, n, rho_p):
        self.gsd = gsd
        self.n = n
        self.rho_p = rho_p * 1e-3
        self.residual = self._residual()

    def _norminv(self, T):
        return t / (self.n - self.residual) + self.residual

    def _norm(self, t):
        return (t - self.residual) / (self.n - self.residual)

    # return kPa
    def _psi(self, r):
        e = self.n / (1. - self.n)
        return Cc / r * np.sqrt(3. / (2. * e) * (1 / (
            2. * np.pi * self.rho_p) * self.gsd.mass(r * 1e-3) / r**2)**
                                (alpha - 1.))

    def _residual(self):
        def _fp(x):
            return self._psi(x) - 1e6

        a = np.power(2.0, -22)
        b = np.power(2.0, -2)
        r = bisect(_fp, a, b)
        return self._theta(r)

    def _theta(self, r):
        return self.n * self.gsd.masscum(r * 1e-3)

    def fit(self, model="vg"):
        r = np.logspace(-20, 6, num=100, base=2.0) * 1e3
        h = pressurehead(self._psi(r))
        t = self._norm(self._theta(r))
        mask = (t >= 0) & (t <= 1)
        m = vgmodel if model == "vg" else fxmodel
        m.params, _ = curve_fit(
            m._Theta,
            h[mask],
            t[mask],
            bounds=m.bounds,
            p0=m.params,
            method="dogbox")
        return m
