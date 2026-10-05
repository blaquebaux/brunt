#!/usr/bin/python3
# =============================================================================
# _brunt_common.py — shared helpers for Blaque Baux BRUNT (the "drift-regime factor" skeptic test).
# Alpaca SIP daily bars (adjustment=all → total return); reads ALPACA_KEY_ID / ALPACA_SECRET_KEY. Read-only.
#
# A Nov-2025 paper claims a cross-sectional equity factor (value + short-term reversal, applied ONLY during
# stock-specific "drift regimes" = >60% positive days in a trailing 63-day window) with an OUT-OF-SAMPLE
# Sharpe > 13 (158% ann, -11.9% maxDD, factor R² < 3%). A Sharpe of 13 is ~5x Medallion's gross — a number that
# almost always means look-ahead, zero costs, or micro-cap fills, not a tradable edge. BRUNT is the family's
# extraordinary-claim test: we can't reproduce the exact setup (no fundamentals here for "value", and Alpaca
# starts 2016 vs their 2004), so we build a faithful PRICE-ONLY analog under honest causal rules WITH costs, and
# ask two questions: (1) does anything REMOTELY near Sharpe-13 appear? (2) is "drift regime" just a momentum
# filter wearing a new name — i.e. is the claimed near-zero factor R² credible? Extraordinary claims, honest test.
# Value proxy = long-term reversal (cheap = poor 3y→1y past return, the standard price-only value stand-in).
# =============================================================================
import os, json, urllib.request, math
import numpy as np

H = {"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]}
START, END = "2016-01-01", "2026-08-01"
_cache = {}

def bars(s):
    if s in _cache: return _cache[s]
    u = (f"https://data.alpaca.markets/v2/stocks/bars?symbols={s}&timeframe=1Day"
         f"&start={START}&end={END}&adjustment=all&feed=sip&limit=10000")
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=40))
        _cache[s] = {b["t"][:10]: b["c"] for b in d.get("bars", {}).get(s, [])}
    except Exception:
        _cache[s] = {}
    return _cache[s]

def panel(syms):
    D = {s: bars(s) for s in syms}; D = {s: v for s, v in D.items() if len(v) > 800}
    if not D: return {}, []
    u = list(D); dates = sorted(set.intersection(*[set(D[s]) for s in u]))
    return {s: np.array([D[s][d] for d in dates], float) for s in u}, dates

def rets(px): return px[1:] / px[:-1] - 1

def stats(r):
    r = np.asarray(r, float); r = r[np.isfinite(r)]
    if len(r) < 30 or r.std() == 0: return dict(sh=float('nan'), cagr=float('nan'), dd=float('nan'), vol=float('nan'))
    cum = np.cumprod(1 + r)
    return dict(sh=r.mean()/r.std()*math.sqrt(252), cagr=cum[-1]**(252/len(r))-1,
                dd=(cum/np.maximum.accumulate(cum)-1).min(), vol=r.std()*math.sqrt(252))

def rankpct(x):
    """Cross-sectional percentile rank in [0,1], NaNs ignored (left at NaN)."""
    x = np.asarray(x, float); out = np.full(len(x), np.nan); m = np.isfinite(x)
    if m.sum() < 3: return out
    order = np.argsort(np.argsort(x[m])); out[m] = order / (m.sum() - 1)
    return out
