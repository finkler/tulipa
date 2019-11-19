import numpy as np

from scipy.stats import lognorm
from tulipa.stats import loglap, rmsd

iso14688_1_2002 = np.array([
    0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0, 630.0,
    2000.0
])


def phi(d):
    return -np.log2(d)


class ps_distribution:
    def __init__(self, sieves):
        if sieves.ndim == 1:
            self.sieves = np.column_stack((iso14688_1_2002[:sieves.size],
                                           sieves))
        else:
            self.sieves = self._normalized(sieves[:, 0], sieves[:, 1])

        a = lognorm.fit(self.sieves)
        b = loglap.fit(self.sieves)
        rms1 = rmsd(lognorm.pdf(self._ps(), *a), self._pp())
        rms2 = rmsd(loglap.pdf(self._ps(), *b), self._pp())
        if rms1 < rms2:
            self._rv = lognorm
            self._shapes = a
        else:
            self._rv = loglap
            self._shapes = b

        self.grading = self._grading()
        self.sort = self._sort()

    def _grading(self):
        def _sym(s, p):
            if p <= .01:
                return ''
            s = s.lower()
            if p <= .05:
                s = '(v{})'.format(s)
            elif p <= .2:
                s = '({})'.format(s)
            return s

        c = self._pp()[0]
        si = self._pp()[1:4].sum()
        s = self._pp()[4:7].sum()
        m = si + c
        g = self._pp()[7:].sum()
        if g > .01:
            u = [('G', g), ('S', s), ('M', m)]
        else:
            u = [('S', s), ('SI', si), ('C', c)]
        u = sorted(u, key=lambda p: -p[1])
        return _sym(*u[2]) + _sym(*u[1]) + u[0][0]

    def _normalized(self, d, w):
        n = iso14688_1_2002.size - 1
        while n > 0 and np.amax(d) > iso14688_1_2002[n]:
            n -= 1
        ps = iso14688_1_2002[:n + 1]
        pp = np.zeros(ps.size)
        for i in range(d.size):
            for j in range(ps.size):
                if d[i] < ps[j]:
                    pp[j] += w[i]
                    break
        return np.column_stack((ps, pp))

    def _ps(self):
        return self.sieves[:, 0]

    def _pp(self):
        return self.sieves[:, 1]

    def _sort(self):
        d95 = phi(self.d(.95))
        d84 = phi(self.d(.84))
        d16 = phi(self.d(.16))
        d5 = phi(self.d(.05))
        si = (d84 - d16) / 4. + (d95 - d5) / 6.6
        if si < .35:
            return 'very well sorted'
        if si < .5:
            return 'well sorted'
        if si < 1.:
            return 'moderately sorted'
        if si < 2.:
            return 'poorly sorted'
        if si < 4.:
            return 'very poorly sorted'
        return 'extremely poorly sorted'

    def d(self, p):
        return self._rv.ppf(p, *self._shapes)

    def mass(self, r):
        return self._rv.pdf(r, *self._shapes)

    def masscum(self, r):
        return self._rv.cdf(r, *self._shapes)
