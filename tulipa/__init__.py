from flask import Flask, Response, request
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

import numpy as np

from tulipa.soil import cneap, flow, sediment


def fit(sieves):
    if loglap.fit(sieves) < lognorm.fit(sieves):
        return loglap
    return lognorm


def header(swcc, hc, cls):
    return "#"


def process(arr, models=[]):
    key = arr.pop(0)
    pp = np.array([float(n) for n in arr[:10]])
    gsd = sediment.ps_distribution(pp)
    rho_b = float(arr[10])
    rho_p = float(arr[11])
    n = 1. - rho_b / rho_p

    model = models[0]
    result = ["{:>6s}".format(key)]
    if len(model) > 0:
        ap = cneap.CNEAP(gsd, n, rho_p)
        result.append("{:9.4f}".format(n))
        result.append("{:9.4f}".format(ap.residual))
        for m in model:
            swcc = ap.fit(model=m)
            for p in swcc.params:
                result.append("{:14.4f}".format(p))

    model = models[1]
    for m in model:
        K = flow.estimate(m, n, gsd.d)
        result.append("{:18.4f}".format(K))

    model = models[2]
    if len(model) > 0:
        c = gsd.grading
        result.append("{:>13s}".format(c))
    return "".join(result)


@app.route('/execute', methods=['POST'])
def execute():
    models = [
        request.form.getlist('swcc'),
        request.form.getlist('hc'),
        request.form.getlist('class')
    ]

    buf = [header(*models)]
    fs = request.files['data-file']
    for line in fs:
        s = line.decode('utf-8').strip()
        if s.startswith('#'):
            continue
        result = process(s.split(), models=models)
        buf.append(result)
    return Response('\r\n'.join(buf), mimetype='text/plain')
