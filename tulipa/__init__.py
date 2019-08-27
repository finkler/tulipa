from flask import Flask, Response, request
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './'

import numpy as np

from tulipa.soil import cneap, sediment
from tulipa import stats

def process(arr, models=[]):
    key = arr.pop(0)
    pp = np.array([float(n) for n in arr[:10]])
    sieves = np.column_stack((sediment.grain_sizes, pp))
    rvc = stats.generic_fit(sieves)
    rho_b = float(arr[10])
    rho_p = float(arr[11])
    n = 1. - rho_b / rho_p
    ap = cneap.CNEAP(rvc, n, rho_p)
    swcc, perr = ap.fit()
    c = sediment.classified(pp)
    fmt = '{:>6s}{:>12s}' + ('{:9.4f}' * (len(swcc.params) + 2))
    return fmt.format(key, c, n, ap.residual, *swcc.params)


@app.route('/execute', methods=['POST'])
def execute():
    models = request.form.getlist('model')
    fs = request.files['data-file']

    fmt = '#{:>5s}{:>12s}' + ('{:>9s}' * 4)
    r = [fmt.format('Name', 'class', 'theta_s', 'theta_r', 'vga', 'vgn')]

    for line in fs:
        s = line.decode('utf-8').strip()
        if s.startswith('#'):
            continue
        r.append(process(s.split(), models=models))
    return Response('\r\n'.join(r), mimetype='text/plain')
