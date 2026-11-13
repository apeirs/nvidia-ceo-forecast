# NVIDIA-CEO Polymarket minute-price forecaster

Entry for make.fun `polymarket-nvidia-ceo-forecast-01`. `main.py` (Python 3.13, standard library only) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

    mkdir -p out && MAKEFUN_INPUT_DIR=../kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py

## Model
For each market, from its last price x and a recent volatility s (a blend of the RMS 1-minute change over the
last 120 minutes and the per-minute RMS 10-minute change over the last 240 minutes, plus a small term proportional
to the price and a floor), the quantile at level tau for minute h after the checkpoint is `x + s * C[tau] * h ** P[tau]`, clamped to
[0, 1] and made non-decreasing. C and P are a handful of global constants fitted on a pooled, scale-normalised
rolling backtest of the event's own history; they are upward-skewed (near-zero markets jump up more often than they
fall). Deterministic, no network, no data files.

Backtest on 50 rolling 4-hour checkpoints of the event's history (2026-11-12 04:00Z to 2026-11-13 04:30Z): mean
pinball loss 0.285 vs 0.389 for the house baseline; 0.265 vs 0.364 on the practice cases. Runs in about 0.1 s per
full-size case.
