import numpy as np

from scipy.optimize import curve_fit, newton
from tulipa.soil.swcc import fxmodel_gen, vgmodel_gen

alpha = 1.38
Cc = 130.0  # um-kPa


class CNEAP:
    def __init__(self, rvc, n, rho_p):
        self.rvc = rvc
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
            2. * np.pi * self.rho_p) * self.rvc.pdf(r) / r**2)**(alpha - 1.))

    def _residual(self, r0=1e-6):
        def _fp(x):
            return self._psi(x) - 1e6

        r = newton(_fp, r0)
        return self._theta(r)

    def _theta(self, r):
        return self.n * self.rvc.cdf(r)

    def fit(self, model="vg", r=None):
        if r is None:
            r = np.logspace(-20, 6, num=100, base=2.0) * 1e3
        h = self._psi(r)
        t = self._norm(self._theta(r))
        mask = (t >= 0) & (t <= 1)
        models = {"fx": fxmodel_gen(), "vg": vgmodel_gen()}
        m = models[model]
        m.params, pcov = curve_fit(
            m._Theta,
            h[mask],
            t[mask],
            bounds=m.bounds,
            p0=m.params,
            jac=m._jac,
            method="dogbox")
        return (m, np.sqrt(np.abs(np.diag(pcov)).mean()))
