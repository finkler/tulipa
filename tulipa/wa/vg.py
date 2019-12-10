import numpy as np


def kr(t, a, n):
    m = 1. - 1. / n
    return np.sqrt(t) * (1. - (1. - t**(1. / m))**m)**2


def psi(t, a, n):
    m = 1. - 1. / n
    me = a * (.046 * m + 2.07 * m**2 + 19.5 * m**3) / (
        1. + 4.7 * m + 16. * m**2)
    _psi = 1. / a * (t**(1 / m) - 1.)**(1 / n)
    return np.max((_psi, me))


def se(h, a, n):
    m = 1. - 1. / n
    return (1. + (a * h)**n)**(-m)


# def _jac(h, a, n):
#     h = np.asarray(h)
#     scalar_input = False
#     if h.ndim == 0:
#         h = h[None]
#         scalar_input = True

#     j = np.zeros((h.size, 2))
#     m = 1. - 1. / n
#     xi = 1. / (1. + np.power(a * h, n))
#     j[:, 0] = -h * np.power(a * h, n - 1.) * np.power(xi, 2. - 1. / n) * m * n
#     j[:, 1] = np.power(xi, m) * (-np.power(a * h, n) * m * np.log(a * h) /
#                                  (1. + np.power(a * h, n)) + np.log(xi) / n**2)

#     if scalar_input:
#         return np.squeeze(j)
#     return j
