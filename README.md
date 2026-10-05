# Blaque Baux Brunt

**The "drift-regime factor" skeptic test — a claimed Sharpe > 13 meets honest causal rules and costs.**

A Nov-2025 paper claims a cross-sectional equity factor (value + short-term reversal, applied *only* during
stock-specific "drift regimes" = >60% positive days in a trailing 63-day window) with an **out-of-sample Sharpe > 13**
(158%/yr, −11.9% maxDD, factor R² < 3%). A Sharpe of 13 is roughly **5× Medallion's gross** — a book that essentially
never has a losing period — which almost always means look-ahead, zero costs, or micro-cap fills. Brunt is the family's
extraordinary-claim test: build a faithful **price-only analog** under honest causal rules *with* costs and see whether
anything near the claim appears, and whether "drift regime" is just a momentum filter wearing a new name.

> **Not investment advice.** Educational/research software. See [DISCLAIMER](DISCLAIMER.md) and [LICENSE](LICENSE).

```bash
python3 research/brunt_1_driftregime.py   # needs Alpaca data keys in the environment
```

## The finding (implausible / not reproducible)

[`research/brunt_1_driftregime.py`](research/brunt_1_driftregime.py) — ~90 liquid names, weekly decile long/short,
value proxied by long-term reversal (no fundamentals available), causal, costed.

| drift-regime factor (price-only analog) | Sharpe | ann. ret | maxDD |
|---|---|---|---|
| **CLAIMED** (2004–24 S&P 500) | **> 13** | ~158% | −11.9% |
| GROSS | **−0.46** | −4.9% | −36% |
| NET @ 1bp/side | −0.49 | −5.3% | −37% |
| NET @ 5bp/side (*the paper's own cost*) | −0.62 | −6.4% | −40% |

- **Nothing near 13** — the honest analog *loses money*.
- **The momentum is in the gate, not the factor.** The drift *regime* is a momentum screen: in-drift names average
  momentum percentile **0.65 vs 0.48** out ("> 60% up-days in 63d" = "the stock is trending up"). Yet the factor's
  *returns* regress on a momentum book at only **R² 1%** — *consistent* with the paper's <3%, because value+reversal
  *inside* a momentum gate is contrarian in return-space. So the low factor-R² is real but does nothing to make a
  Sharpe-13 robust — it's a true statistic pointed at the wrong question.
- **The drift filter *hurts*** here (conditional −0.46 vs unconditional −0.02): the regime added turnover, not edge.

## Verdict

**IMPLAUSIBLE / NOT REPRODUCIBLE (price-only OOS).** A Sharpe of 13 is an extraordinary claim, and a faithful causal
analog gets nowhere near it. **Honest limit:** I can't fully audit *their* number — their "value" is fundamental (I
proxy with long-term reversal), their universe is 500 names over 2004–24, mine 93 over 2016+ — so this falsifies
*reproducibility*, not their private code. But the burden of proof sits on the Sharpe-13 side, and nothing here
supports it; treat the figure as a frictionless / look-ahead / selection artifact until an audit with real costs and
fills says otherwise. **The family's standing rule: an extraordinary Sharpe is a red flag, not a prize** — trust the
scorecard, never the headline number.

## Status

**Research / methodology.** An honest skeptic test — faithful where the data allows, explicit about what it can and
cannot conclude. Causal signals, the paper's own cost assumption, momentum-decomposition, Alpaca SIP daily total
return. No live capital.
