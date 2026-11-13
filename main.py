# SPDX-License-Identifier: MIT
"""Minute-price quantile forecaster for make.fun polymarket-nvidia-ceo-forecast-01 (Python 3.13, stdlib only).

For each market: x = last price, s = recent volatility (root mean square of 1-minute changes over the last
WINDOW minutes, floored). Quantile at level tau for minute h: x + s * C[tau] * h ** P[tau], clamped to [0, 1].
C and P are global constants fitted on a pooled, scale-normalised backtest (upward-skewed: near-zero markets
spike up more often than they fall)."""
import json
import math
import os

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

PARAMS = {
    'window': 120,          # minutes of 1-minute changes for the volatility estimate
    'floor': 0.0002,        # minimum volatility (price units per minute)
    'short_sigma': 0.0025,  # volatility assumed when there are fewer than 'min_n' changes
    'min_n': 30,
    'c': [-0.95, -0.23, 0.0, 0.034, 2.2],
    'p': [0.5, 0.5, 0.5, 0.6, 0.3],
}


def clean_prices(raw):
    out = []
    last = None
    for v in raw if isinstance(raw, list) else []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float('nan')
        if not math.isfinite(f):
            if last is None:
                continue
            f = last
        f = min(1.0, max(0.0, f))
        out.append(f)
        last = f
    return out


def volatility(p, P):
    w = P['window']
    seg = p[-w - 1:]
    n = len(seg) - 1
    if n <= 0:
        return P['short_sigma']
    ss = 0.0
    for a, b in zip(seg, seg[1:]):
        ss += (b - a) * (b - a)
    s = math.sqrt(ss / n)
    if n < P['min_n']:
        # blend toward the default when the history is short
        k = n / P['min_n']
        s = k * s + (1 - k) * P['short_sigma']
    return max(s, P['floor'])


def forecast_market(prices, horizon=240, P=PARAMS):
    p = clean_prices(prices)
    if not p:
        return [[0.0, 0.0, 0.0, 0.0, 0.0] if False else [0.0, 0.005, 0.02, 0.05, 0.2] for _ in range(horizon)]
    x = p[-1]
    s = volatility(p, P)
    c, pw = P['c'], P['p']
    rows = []
    for h in range(1, horizon + 1):
        row = []
        for j in range(5):
            q = x + s * c[j] * h ** pw[j]
            row.append(min(1.0, max(0.0, q)))
        rows.append(row)
    return rows


def finalize(rows, horizon):
    out = []
    for i in range(horizon):
        row = rows[i] if i < len(rows) else rows[-1]
        r = []
        prev = 0.0
        for v in row:
            v = float(v)
            if not math.isfinite(v):
                v = prev
            v = round(min(1.0, max(0.0, v)), 6) + 0.0
            if v < prev:
                v = prev
            r.append(v)
            prev = v
        out.append(r)
    return out


def safe_rows(prices, horizon):
    x = 0.5
    try:
        p = clean_prices(prices)
        if p:
            x = p[-1]
    except Exception:
        pass
    return [[x + z * 0.0025 * math.sqrt(h) for z in (-1.644854, -0.67449, 0.0, 0.67449, 1.644854)]
            for h in range(1, horizon + 1)]


def run_case(case):
    horizon = int(case.get('horizon_minutes', 240) or 240)
    fc = []
    for m in case.get('markets', []):
        prices = m.get('prices', [])
        try:
            rows = finalize(forecast_market(prices, horizon), horizon)
        except Exception:
            rows = finalize(safe_rows(prices, horizon), horizon)
        fc.append({'id': m['id'], 'q': rows})
    return {'forecasts': fc}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fn in sorted(os.listdir(INPUT_DIR)):
        if not fn.endswith('.in'):
            continue
        with open(os.path.join(INPUT_DIR, fn), encoding='utf-8') as f:
            case = json.load(f)
        out = run_case(case)
        with open(os.path.join(OUTPUT_DIR, fn[:-3] + '.out'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(out, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
