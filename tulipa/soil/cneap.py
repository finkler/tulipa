import numpy as np

from scipy.optimize import bisect, curve_fit
from tulipa.wa import vg

alpha = 1.38
Cc = 130.0  # um-kPa


# kPa to cm
def ph(psi):
    g = 9.81
    rho = 997.0
    return (psi * 1e3 / (g * rho)) * 100.0


class CNEAP:
    def __init__(self, psd, n=None, rho_p=2.65):
        self._psd = psd
        self.n = n if n is not None else psd.porosity
        self.rho_p = rho_p * 1e-3
        self.r = self._residual()

    def _eff(self, t):
        return (t - self.r) / (self.n - self.r)

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
        return self.n * self.psd.masscum(r * 1e-3)

    def model(self):
        r = np.logspace(-20, 6, num=100, base=2.0) * 1e3
        h = ph(self._psi(r))
        t = self._eff(self._theta(r))
        bounds = ([0, 1], np.inf)
        mask = (t >= 0) & (t <= 1)
        p0 = [0.005, 2.0]
        popt, _ = curve_fit(
            vg.se, h[mask], t[mask], bounds=bounds, p0=p0, method="dogbox")
        return (self.r,) + popt
