from flask import Flask, Response, request
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'

import numpy as np

from tulipa.soil.cneap import CNEAP
from tulipa.soil.sediment import ps_distribution


def header(swcc, hc, cls):
    return "#"


def process(arr, models=[]):
    key = arr.pop(0)
    pp = np.array([float(n) for n in arr[:10]])
    psd = ps_distribution(pp)
    rho_b = float(arr[10])
    rho_p = float(arr[11])
    n = 1. - rho_b / rho_p

    r, vga, vgn = CNEAP(psd, n, rho_p).model()
    result = [
        "{:>6s}".format(key),
        "{:9.4f}".format(n),
        "{:9.4f}".format(r),
        "{:9.4f}".format(vga),
        "{:9.4f}".format(vgn)
    ]
    return "".join(result)


@app.route('/tulipa', methods=['POST'])
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
