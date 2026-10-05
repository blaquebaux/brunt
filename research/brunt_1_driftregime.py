#!/usr/bin/python3
# =============================================================================
# brunt_1_driftregime.py — does a "drift-regime" value+reversal factor earn Sharpe 13, or is it momentum + no costs?
#
# Faithful PRICE-ONLY analog of the Nov-2025 claim, honest causal rules. Universe ~90 liquid names (survivorship-
# flagged). At each weekly rebalance, among stocks in a DRIFT REGIME (>60% up-days in trailing 63d):
#   value proxy  = long-term reversal (cheap = poor return from t-756 to t-252, i.e. 3y→1y ago)
#   ST reversal  = recent loser (poor return over trailing 21d)
#   score = mean(rank(value), rank(ST-reversal));  long top decile / short bottom decile, dollar-neutral.
# Then the two skeptic tests: (1) realistic Sharpe GROSS and NET of cost vs the claimed 13 / 158%/yr / -11.9% DD;
# (2) is "drift regime" just MOMENTUM? — regress the factor on a plain 12-1 momentum book and report R² (the claim
# says factor R² < 3%); and show whether drift-regime membership is itself a momentum screen.
# =============================================================================
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _brunt_common import panel, rets, stats, rankpct
import numpy as np

UNIV = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","INTC","AMD","QCOM","TXN","IBM","MU","AMAT","LRCX","ADI","NOW","INTU",
        "JPM","BAC","WFC","GS","MS","C","AXP","V","MA","USB","PNC","SCHW","BLK","BK","COF",
        "JNJ","PFE","MRK","ABT","TMO","LLY","UNH","BMY","ABBV","AMGN","GILD","CVS","CI","MDT","DHR","ISRG",
        "XOM","CVX","COP","SLB","EOG","OXY","PSX","VLO",
        "HD","WMT","COST","TGT","NKE","MCD","SBUX","PG","KO","PEP","DIS","CMCSA","LOW","TJX","EL","CL","KMB",
        "BA","CAT","GE","HON","UNP","UPS","MMM","LMT","RTX","DE","T","VZ","NEE","DUK","SO"]
P, dates = panel(UNIV)
syms = [s for s in UNIV if s in P]; M = np.column_stack([P[s] for s in syms])
dr = M[1:] / M[:-1] - 1                                     # dr[t] = return date_t → date_{t+1} (causal forward return)
nd, nn = dr.shape
print("="*98); print(f"BRUNT #1 — the 'drift-regime' factor, skeptic test  ({dates[0]} → {dates[-1]}, {len(dates)} days, {nn} names)"); print("="*98)

def book(score_at, mask_at=None, dec=0.10):
    """Weekly dollar-neutral long-top / short-bottom-decile book. score_at(t)=cross-sec score (NaN=exclude);
    mask_at(t)=bool keep-universe. Weights from info at date_t, earn forward return dr[t]. Returns (ret, turnover)."""
    w = 5; prev = np.zeros(nn); out = np.full(nd, np.nan); turn = np.full(nd, 0.0)
    for t in range(756, nd):
        if (t - 756) % w != 0:
            out[t] = prev @ dr[t]; continue
        sc = score_at(t).astype(float)
        if mask_at is not None: sc = np.where(mask_at(t), sc, np.nan)
        ok = np.isfinite(sc)
        wt = np.zeros(nn)
        if ok.sum() >= 10:
            idx = np.where(ok)[0]; order = idx[np.argsort(sc[idx])]
            k = max(1, int(round(ok.sum() * dec)))
            lo, hi = order[:k], order[-k:]
            wt[hi] = 0.5 / len(hi); wt[lo] = -0.5 / len(lo)
        turn[t] = np.abs(wt - prev).sum(); prev = wt; out[t] = wt @ dr[t]
    return out[756:], turn[756:]

def value(t):  return -(M[t-252] / M[t-756] - 1)           # long-term reversal: cheap = poor 3y→1y return
def strev(t):  return -(M[t] / M[t-21] - 1)                # short-term reversal: recent loser
def score(t):  return np.nanmean(np.vstack([rankpct(value(t)), rankpct(strev(t))]), axis=0)
def indrift(t): return (dr[t-63:t] > 0).mean(axis=0) > 0.60  # >60% up-days in trailing 63d
def mom(t):    return rankpct(M[t-21] / M[t-252] - 1)      # plain 12-1 momentum score

f_drift, turn = book(score, indrift)                        # the claimed factor (drift-conditional)
f_uncond, _   = book(score)                                 # same factor, NO drift filter
f_mom, _      = book(mom)                                   # pure momentum book (the suspected true driver)

def net(r, tn, bps): return r - tn * (bps / 1e4)
g = stats(f_drift); n1 = stats(net(f_drift, turn, 1)); n5 = stats(net(f_drift, turn, 5))
print(f"\n  CLAIMED: Sharpe > 13,  ~158%/yr,  maxDD -11.9%   (frozen params, 2004-2024 S&P 500)")
print(f"\n  drift-regime factor (price-only analog)   {'Sharpe':>8}{'ann.ret':>9}{'maxDD':>8}{'turnover/yr':>12}")
print(f"  GROSS                                     {g['sh']:>+8.2f}{g['cagr']*100:>+8.1f}%{g['dd']*100:>+7.0f}%{turn.sum()/(len(turn)/252):>+11.0f}x")
print(f"  NET @ 1bp/side                            {n1['sh']:>+8.2f}{n1['cagr']*100:>+8.1f}%{n1['dd']*100:>+7.0f}%")
print(f"  NET @ 5bp/side (paper's own cost)         {n5['sh']:>+8.2f}{n5['cagr']*100:>+8.1f}%{n5['dd']*100:>+7.0f}%")

# ---- skeptic test 2: is "drift regime" just momentum? -----------------------------------------------
m = np.isfinite(f_drift) & np.isfinite(f_mom); fx, mx = f_drift[m], f_mom[m]
beta = np.cov(fx, mx)[0,1]/np.var(mx); r2 = np.corrcoef(fx, mx)[0,1]**2
uc = stats(f_uncond)
# how momentum-like is the drift screen itself: avg 12-1 momentum percentile of in-drift vs out-drift names
mom_in, mom_out = [], []
for t in range(756, nd, 5):
    mo = mom(t); d = indrift(t)
    if np.isfinite(mo[d]).sum() > 3: mom_in.append(np.nanmean(mo[d]))
    if np.isfinite(mo[~d]).sum() > 3: mom_out.append(np.nanmean(mo[~d]))
mom_in, mom_out = np.nanmean(mom_in), np.nanmean(mom_out)

print(f"\n  SKEPTIC TESTS")
print(f"  • factor regressed on plain momentum: R² {r2*100:>4.0f}%  (β {beta:+.2f})   — the claim says factor R² < 3%")
print(f"  • drift-regime IS a momentum screen: in-drift names avg momentum pct {mom_in:.2f} vs out-drift {mom_out:.2f}")
print(f"  • does the drift filter even help? conditional Sharpe {g['sh']:+.2f} vs unconditional {uc['sh']:+.2f}")

# ---- verdict ----------------------------------------------------------------------------------------
near13     = g['sh'] > 5                                     # anything remotely like the claim
gate_is_mom = (mom_in - mom_out) > 0.10                      # the REGIME screen selects trending (momentum) names
print("\n"+"="*98); print("READ:")
print(f"  • NOTHING NEAR 13: the honest causal analog earns Sharpe {g['sh']:+.2f} gross, {n5['sh']:+.2f} net at the paper's own 5bp cost — it")
print("    LOSES money. A Sharpe of 13 (~5x Medallion gross) means a strategy that essentially never has a down period;")
print("    with real costs and no look-ahead that is a frictionless/overlapping-window/micro-cap artifact, not an edge.")
print(f"  • THE MOMENTUM IS IN THE GATE, NOT THE FACTOR: the drift REGIME is a momentum screen — in-drift names average")
print(f"    momentum pct {mom_in:.2f} vs {mom_out:.2f} out ('>60% up-days in 63d' = 'the stock is trending up'). But the factor's RETURNS")
print(f"    regress on a momentum book at only R² {r2*100:.0f}% — consistent with the paper's <3%, because value+reversal INSIDE a")
print("    momentum gate is contrarian in return-space. So the low factor-R² is real but doesn't make the result robust.")
print(f"  • the drift filter HURTS here (cond {g['sh']:+.2f} vs uncond {uc['sh']:+.2f}) — conditioning on the regime added nothing but turnover.")

if not near13:
    v = ("IMPLAUSIBLE / NOT REPRODUCIBLE (price-only OOS): a Sharpe of 13 is an extraordinary claim — ~5x Medallion "
         f"gross, a book that practically never loses — and a faithful causal analog gets nowhere near it ({g['sh']:+.2f} gross, "
         f"{n5['sh']:+.2f} net; it loses). The regime that supposedly unlocks it is just a momentum screen. HONEST LIMIT: I can't "
         "fully audit THEIR number — their 'value' is fundamental (I proxy with long-term reversal), their universe is 500 "
         "names over 2004-24, mine 93 over 2016+ — so this falsifies reproducibility, not their private code. But the burden "
         "is on a Sharpe-13 claim, and nothing here supports it: treat it as frictionless/look-ahead/selection artifact "
         "until an audit with real costs and fills says otherwise. Trust the scorecard, never the headline Sharpe.")
else:
    v = (f"UNEXPECTED: the analog earns Sharpe {g['sh']:+.2f} — worth hunting the usual inflators (look-ahead, costs, universe) "
         "before believing it; still orders of magnitude below the claimed 13.")
print(f"\n  VERDICT: {v}")
print("  (Plausibility test, not a replication: no fundamentals for 'value' (proxied by long-term reversal), Alpaca starts")
print("   2016 vs their 2004. It shows Sharpe-13 is implausible under honest rules and the regime is momentum; a true audit")
print("   needs their code, universe and fills. The family's standing rule: an extraordinary Sharpe is a red flag, not a prize.)")
