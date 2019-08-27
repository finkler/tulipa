import numpy as np

grain_sizes = np.array(
    [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0])


def normalized(d, w):
    ps = grain_sizes
    pp = np.zeros(ps.size)
    for i in range(d.size):
        for j in range(ps.size):
            if d[i] < ps[j]:
                pp[j] += w[i]
                break
    return np.column_stack((ps, pp))


def phi(d):
    return -np.log2(d)


def sorting(d):
    d95 = phi(d(.95))
    d84 = phi(d(.84))
    d16 = phi(d(.16))
    d5 = phi(d(.05))
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


def classified(pp):
    def _sym(s, p):
        if p <= .01:
            return ''
        s = s.lower()
        if p <= .05:
            s = '(v{})'.format(s)
        elif p <= .2:
            s = '({})'.format(s)
        return s

    c = pp[0]
    si = pp[1:4].sum()
    s = pp[4:7].sum()
    m = si + c
    g = pp[7:].sum()
    if g > .01:
        u = [('G', g), ('S', s), ('M', m)]
    else:
        u = [('S', s), ('SI', si), ('C', c)]
    u = sorted(u, key=lambda p: -p[1])
    return _sym(*u[2]) + _sym(*u[1]) + u[0][0]
