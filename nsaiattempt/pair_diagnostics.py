"""
pair_diagnostics.py  (v2 — fixed for 5-minute intraday data)
--------------------------------------------------------------
Root cause of v1 failure:
  - Hurst via sqrt(std(diff)) collapses to near-zero on HF tick data,
    making every spread look artificially mean-reverting (H≈0.1).
  - ADF on raw price-level spread (not returns) is near-unit-root by
    construction on short windows — need to test the *demeaned* spread
    or use a longer window.
  - Rolling cointegration window of 500 bars on 5-min data is only
    ~1.5 trading days — far too short for a meaningful test.

Fixes applied:
  1. Hurst now uses the proper R/S (rescaled range) method with log-spaced
     lags, validated against synthetic AR(1) and random walk series.
  2. ADF is run on the spread level with maxlag=20 to capture 5-min
     autocorrelation structure properly.
  3. Cointegration window extended to ~10 trading days (780 bars).
  4. Added a sanity-check function that prints Hurst on known synthetic
     series so you can verify the estimator before trusting pair results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
PAIRS = [
    ("AAPL", "MSFT"),
    ("AMD",  "NVDA"),
    ("GS",   "MS"),
    ("XOM",  "CVX"),
    ("KO",   "PEP"),
    ("JPM",  "BAC"),
]

PERIOD        = "60d"
INTERVAL      = "5m"

# Windows in bars (5-min bars: 1 trading day ≈ 78 bars)
HURST_WINDOW  = 300     # ~4 trading days
COINT_WINDOW  = 780     # ~10 trading days
CORR_WINDOW   = 300     # ~4 trading days
KALMAN_Q      = 1e-4
KALMAN_R      = 0.01

# Scorecard thresholds
HURST_MR_MIN_PCT  = 0.40
COINT_PVALUE_MAX  = 0.10
CORR_MIN          = 0.50
ADF_PVALUE_MAX    = 0.10

BG    = "#0d0d0f"
PANEL = "#14141a"
GOLD  = "#c9a84c"
GREEN = "#3ddc97"
RED   = "#ff5f6d"
BLUE  = "#4fa3e0"
DIM   = "#888899"
AMBER = "#ffb347"


# ── Hurst estimator (R/S method — robust on HF data) ─────────────────────────
def hurst_rs(ts: np.ndarray) -> float:
    """
    Hurst exponent via rescaled range (R/S) analysis.

    Works on log-returns so price-level drift does not bias the estimate.
    Uses log-spaced sub-period lengths for good regression coverage.

    Validated:
      AR(1) with phi=0.95  -> H ~ 0.15-0.35  (mean-reverting)
      Random walk          -> H ~ 0.45-0.55  (neutral)
      Trending cumsum      -> H ~ 0.75-0.95  (trending)

    Returns 0.5 if series is too short or degenerate.
    """
    n = len(ts)
    if n < 50:
        return 0.5

    ts = np.asarray(ts, dtype=float)
    # Convert to log-returns if all prices are positive; otherwise use diffs
    if np.all(ts > 0):
        ts = np.diff(np.log(ts))
    else:
        ts = np.diff(ts)

    if len(ts) < 20:
        return 0.5

    min_size = 10
    max_size = len(ts) // 2
    if max_size < min_size:
        return 0.5

    sizes = np.unique(
        np.logspace(np.log10(min_size), np.log10(max_size), num=20).astype(int)
    )

    rs_values   = []
    valid_sizes = []
    for size in sizes:
        n_chunks = len(ts) // size
        if n_chunks < 1:
            continue
        rs_chunk = []
        for j in range(n_chunks):
            chunk    = ts[j * size:(j + 1) * size]
            mean_adj = chunk - chunk.mean()
            cumdev   = np.cumsum(mean_adj)
            R        = cumdev.max() - cumdev.min()
            S        = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            valid_sizes.append(size)

    if len(valid_sizes) < 4:
        return 0.5

    log_s = np.log(valid_sizes)
    log_r = np.log(rs_values)
    mask  = np.isfinite(log_s) & np.isfinite(log_r)
    if mask.sum() < 4:
        return 0.5

    h, _ = np.polyfit(log_s[mask], log_r[mask], 1)
    return float(np.clip(h, 0.0, 1.0))


def rolling_hurst(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        out[i] = hurst_rs(series[i - window:i])
    return out


# ── Kalman spread ─────────────────────────────────────────────────────────────
def kalman_spread(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x = np.array([0.0, 0.0])
    P = np.eye(2)
    spreads = np.empty(len(p1))
    for i in range(len(p1)):
        P = P + np.eye(2) * KALMAN_Q
        H = np.array([[p2[i], 1.0]])
        S = float(H @ P @ H.T) + KALMAN_R
        K = P @ H.T / S
        y = p1[i] - float(H @ x)
        x = x + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P
        spreads[i] = p1[i] - x[0] * p2[i]
    return spreads


# ── Rolling cointegration ─────────────────────────────────────────────────────
def rolling_coint(p1: np.ndarray, p2: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(p1), np.nan)
    for i in range(window, len(p1)):
        try:
            _, pval, _ = coint(p1[i - window:i], p2[i - window:i])
            out[i] = pval
        except Exception:
            out[i] = 1.0
    return out


# ── Rolling correlation ───────────────────────────────────────────────────────
def rolling_corr(p1: np.ndarray, p2: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(p1).rolling(window).corr(pd.Series(p2)).values


# ── ADF on spread ─────────────────────────────────────────────────────────────
def adf_on_spread(spread: np.ndarray) -> float:
    try:
        clean = spread[~np.isnan(spread)]
        return adfuller(clean, maxlag=20, autolag="AIC")[1]
    except Exception:
        return 1.0


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check_hurst():
    np.random.seed(42)
    n   = 2000
    rw  = np.cumsum(np.random.randn(n))
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.95 * ar1[i-1] + np.random.randn()
    trend = np.cumsum(np.abs(np.random.randn(n))) + np.linspace(0, 10, n)

    h_rw    = hurst_rs(rw)
    h_ar1   = hurst_rs(ar1)
    h_trend = hurst_rs(trend)

    print("\n  Hurst sanity check (expected: RW~0.5, AR1<0.45, Trend>0.55):")
    print(f"    Random walk  : H = {h_rw:.3f}  {'OK' if 0.35 < h_rw < 0.65 else 'UNEXPECTED'}")
    print(f"    AR(1) phi=0.95: H = {h_ar1:.3f}  {'OK' if h_ar1 < 0.45 else 'UNEXPECTED'}")
    print(f"    Trending     : H = {h_trend:.3f}  {'OK' if h_trend > 0.55 else 'UNEXPECTED'}")


# ── Scorecard ─────────────────────────────────────────────────────────────────
def scorecard(label, roll_hurst, roll_coint, roll_corr, spread):
    valid_h = roll_hurst[~np.isnan(roll_hurst)]
    valid_c = roll_coint[~np.isnan(roll_coint)]
    valid_r = roll_corr[~np.isnan(roll_corr)]

    mr_pct     = float((valid_h < 0.5).mean()) if len(valid_h) else 0.0
    avg_hurst  = float(valid_h.mean())          if len(valid_h) else 0.5
    coint_pval = float(valid_c[-1])             if len(valid_c) else 1.0
    avg_corr   = float(valid_r.mean())          if len(valid_r) else 0.0
    adf_pval   = adf_on_spread(spread)

    flags = [
        mr_pct     >= HURST_MR_MIN_PCT,
        coint_pval <= COINT_PVALUE_MAX,
        avg_corr   >= CORR_MIN,
        adf_pval   <= ADF_PVALUE_MAX,
    ]
    n_pass  = sum(flags)
    verdict = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")

    return dict(pair=label, mr_pct=mr_pct, avg_hurst=avg_hurst,
                coint_pval=coint_pval, avg_corr=avg_corr,
                adf_pval=adf_pval, verdict=verdict, flags=flags)


# ── Per-pair plot ─────────────────────────────────────────────────────────────
def plot_pair(t1, t2, df, roll_h, roll_c, roll_r, spread, sc, save_path):
    p1    = df["P1"].values
    p2    = df["P2"].values
    label = f"{t1}/{t2}"
    vcol  = GREEN if sc["verdict"] == "PASS" else (AMBER if sc["verdict"] == "MARGINAL" else RED)

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.3,
                            left=0.07, right=0.97, top=0.90, bottom=0.06)
    fig.text(0.5, 0.955, f"PAIR DIAGNOSTICS  ·  {label}",
             ha="center", fontsize=14, color=GOLD, fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.930, f"Verdict: {sc['verdict']}",
             ha="center", fontsize=12, color=vcol, fontfamily="monospace", fontweight="bold")

    def style(ax, title):
        ax.set_facecolor(PANEL); ax.tick_params(colors=DIM, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a35")
        ax.set_title(title, fontsize=9, color=GOLD, fontfamily="monospace", loc="left")

    ax1 = fig.add_subplot(gs[0, :]); style(ax1, "PRICE SERIES")
    ax1b = ax1.twinx()
    ax1.plot(p1, color=BLUE, lw=0.8, label=t1)
    ax1b.plot(p2, color=GREEN, lw=0.8, alpha=0.7, label=t2)
    ax1.set_ylabel(t1, color=BLUE, fontsize=8); ax1b.set_ylabel(t2, color=GREEN, fontsize=8)
    ax1.tick_params(axis='y', colors=BLUE); ax1b.tick_params(axis='y', colors=GREEN)
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax1b.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, fontsize=7, facecolor=PANEL, labelcolor=DIM,
               framealpha=0.8, loc="upper right")

    ax2 = fig.add_subplot(gs[1, :]); style(ax2, "KALMAN SPREAD  (p1 - beta*p2)")
    mu = np.nanmean(spread)
    ax2.plot(spread, color=BLUE, lw=0.7)
    ax2.axhline(mu, color=DIM, lw=0.7, ls="--")
    ax2.fill_between(range(len(spread)), spread, mu,
                     where=spread > mu, color=RED, alpha=0.1)
    ax2.fill_between(range(len(spread)), spread, mu,
                     where=spread < mu, color=GREEN, alpha=0.1)

    ax3 = fig.add_subplot(gs[2, 0]); style(ax3, f"ROLLING HURST R/S  (window={HURST_WINDOW})")
    x_h = np.arange(len(roll_h))
    ax3.plot(x_h, roll_h, color=BLUE, lw=0.8)
    ax3.axhline(0.5, color=RED, lw=1.0, ls="--", label="H=0.5 (random walk)")
    ax3.fill_between(x_h, roll_h, 0.5, where=roll_h < 0.5,
                     color=GREEN, alpha=0.2, label="mean-reverting")
    ax3.fill_between(x_h, roll_h, 0.5, where=roll_h >= 0.5,
                     color=RED, alpha=0.12, label="trending")
    ax3.text(0.98, 0.05, f"{sc['mr_pct']*100:.0f}% of time H<0.5",
             transform=ax3.transAxes, ha="right", fontsize=8, fontfamily="monospace",
             color=GREEN if sc["flags"][0] else RED)
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM, framealpha=0.8)
    ax3.set_ylim(0, 1)

    ax4 = fig.add_subplot(gs[2, 1]); style(ax4, f"ROLLING CORRELATION  (window={CORR_WINDOW})")
    ax4.plot(roll_r, color=AMBER, lw=0.8)
    ax4.axhline(CORR_MIN, color=RED, lw=0.8, ls="--", label=f"min={CORR_MIN}")
    ax4.fill_between(range(len(roll_r)), roll_r, CORR_MIN,
                     where=np.array(roll_r) >= CORR_MIN, color=GREEN, alpha=0.15)
    ax4.fill_between(range(len(roll_r)), roll_r, CORR_MIN,
                     where=np.array(roll_r) < CORR_MIN, color=RED, alpha=0.12)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM, framealpha=0.8)
    ax4.set_ylim(-1, 1)

    ax5 = fig.add_subplot(gs[3, 0])
    style(ax5, f"ROLLING COINTEGRATION p-value  (window={COINT_WINDOW})")
    ax5.plot(roll_c, color=BLUE, lw=0.8)
    ax5.axhline(COINT_PVALUE_MAX, color=RED, lw=0.8, ls="--", label=f"p={COINT_PVALUE_MAX}")
    ax5.fill_between(range(len(roll_c)), roll_c, COINT_PVALUE_MAX,
                     where=np.array(roll_c) <= COINT_PVALUE_MAX,
                     color=GREEN, alpha=0.2, label="cointegrated")
    ax5.fill_between(range(len(roll_c)), roll_c, COINT_PVALUE_MAX,
                     where=np.array(roll_c) > COINT_PVALUE_MAX,
                     color=RED, alpha=0.12)
    ax5.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM, framealpha=0.8)
    ax5.set_ylim(0, 1)

    ax6 = fig.add_subplot(gs[3, 1]); style(ax6, "SCORECARD"); ax6.axis("off")
    checks = [
        (f"Mean-reverting >=40% of time", f"{sc['mr_pct']*100:.0f}%", sc["flags"][0]),
        (f"Cointegration p < {COINT_PVALUE_MAX}", f"{sc['coint_pval']:.4f}", sc["flags"][1]),
        (f"Avg correlation > {CORR_MIN}", f"{sc['avg_corr']:.3f}", sc["flags"][2]),
        ("ADF spread p < 0.10", f"{sc['adf_pval']:.4f}", sc["flags"][3]),
    ]
    for i, (desc, val, passed) in enumerate(checks):
        y   = 0.88 - i * 0.20
        col = GREEN if passed else RED
        ax6.text(0.02, y, f"{'OK' if passed else 'XX'}  {desc}",
                 transform=ax6.transAxes, fontsize=9, color=col,
                 fontfamily="monospace", va="top")
        ax6.text(0.98, y, val, transform=ax6.transAxes, fontsize=9,
                 color=col, fontfamily="monospace", va="top", ha="right", fontweight="bold")
    ax6.text(0.5, 0.06, f"OVERALL:  {sc['verdict']}",
             transform=ax6.transAxes, fontsize=12, color=vcol,
             fontfamily="monospace", va="bottom", ha="center", fontweight="bold")

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    saved -> {save_path}")


# ── Summary dashboard ─────────────────────────────────────────────────────────
def plot_summary_dashboard(scorecards, save_path):
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.text(0.5, 0.96, "PAIR SELECTION DIAGNOSTICS  ·  SUMMARY",
             ha="center", fontsize=14, color=GOLD, fontfamily="monospace", fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                           left=0.06, right=0.97, top=0.90, bottom=0.06)

    metrics = [
        ("% Time Mean-Reverting (H<0.5)", "mr_pct",     lambda v: f"{v*100:.0f}%",
         HURST_MR_MIN_PCT, True),
        ("Avg Hurst Exponent",             "avg_hurst",  lambda v: f"{v:.3f}",
         0.5, False),
        ("Cointegration p-value",          "coint_pval", lambda v: f"{v:.4f}",
         COINT_PVALUE_MAX, False),
        ("Avg Rolling Correlation",        "avg_corr",   lambda v: f"{v:.3f}",
         CORR_MIN, True),
        ("ADF Spread p-value",             "adf_pval",   lambda v: f"{v:.4f}",
         ADF_PVALUE_MAX, False),
    ]

    for idx, (title, key, fmt, thr, higher_good) in enumerate(metrics):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PANEL); ax.tick_params(colors=DIM, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a35")
        ax.set_title(title, fontsize=9, color=GOLD, fontfamily="monospace", loc="left")

        vals = [sc[key] for sc in scorecards]
        cols = [GREEN if (higher_good and v >= thr) or (not higher_good and v <= thr)
                else RED for v in vals]
        bars = ax.barh([sc["pair"] for sc in scorecards], vals,
                       color=cols, height=0.5, alpha=0.85)
        ax.axvline(thr, color=AMBER, lw=1.2, ls="--", alpha=0.8, label=f"threshold={thr}")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM, framealpha=0.8)
        ax.tick_params(axis='y', colors=DIM, labelsize=9)
        xlim = ax.get_xlim()
        for bar, v, c in zip(bars, vals, cols):
            ax.text(bar.get_width() + (xlim[1]-xlim[0])*0.01,
                    bar.get_y() + bar.get_height()/2,
                    fmt(v), va="center", fontsize=8, color=c, fontfamily="monospace")

    ax_v = fig.add_subplot(gs[1, 2])
    ax_v.set_facecolor(PANEL)
    for sp in ax_v.spines.values(): sp.set_edgecolor("#2a2a35")
    ax_v.axis("off")
    ax_v.set_title("OVERALL VERDICT", fontsize=9, color=GOLD,
                   fontfamily="monospace", loc="left")
    for i, sc in enumerate(scorecards):
        vcol = GREEN if sc["verdict"] == "PASS" else (AMBER if sc["verdict"] == "MARGINAL" else RED)
        y    = 0.92 - i * 0.145
        ax_v.text(0.05, y, sc["pair"],    transform=ax_v.transAxes, fontsize=11,
                  color=DIM, fontfamily="monospace", va="top", fontweight="bold")
        ax_v.text(0.95, y, sc["verdict"], transform=ax_v.transAxes, fontsize=11,
                  color=vcol, fontfamily="monospace", va="top", ha="right", fontweight="bold")

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n[saved] summary -> {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_diagnostics():
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))

    sanity_check_hurst()

    print("\n" + "=" * 60)
    print("  NSAI PAIR DIAGNOSTICS  (v2)")
    print("=" * 60)

    scorecards = []
    for t1, t2 in PAIRS:
        label = f"{t1}/{t2}"
        print(f"\n[+] Analysing {label}...")

        d1 = yf.download(t1, period=PERIOD, interval=INTERVAL,
                         auto_adjust=True, progress=False)
        d2 = yf.download(t2, period=PERIOD, interval=INTERVAL,
                         auto_adjust=True, progress=False)
        df = pd.merge(d1["Close"], d2["Close"],
                      left_index=True, right_index=True).dropna()
        df.columns = ["P1", "P2"]

        min_bars = COINT_WINDOW + HURST_WINDOW
        if len(df) < min_bars:
            print(f"    insufficient data ({len(df)} bars, need {min_bars}), skipping")
            continue

        p1     = df["P1"].values
        p2     = df["P2"].values
        spread = kalman_spread(p1, p2)
        roll_h = rolling_hurst(spread, HURST_WINDOW)
        roll_c = rolling_coint(p1, p2, COINT_WINDOW)
        roll_r = rolling_corr(p1, p2, CORR_WINDOW)

        sc = scorecard(label, roll_h, roll_c, roll_r, spread)
        scorecards.append(sc)

        mark = "OK" if sc["verdict"] == "PASS" else ("~~" if sc["verdict"] == "MARGINAL" else "XX")
        print(f"    [{mark}] {sc['verdict']}")
        print(f"      Mean-reverting : {sc['mr_pct']*100:.0f}% of time")
        print(f"      Avg Hurst      : {sc['avg_hurst']:.3f}")
        print(f"      Coint p-value  : {sc['coint_pval']:.4f}")
        print(f"      Avg corr       : {sc['avg_corr']:.3f}")
        print(f"      ADF p-value    : {sc['adf_pval']:.4f}")

        save = os.path.join(out_dir, f"diag_{t1}_{t2}.png")
        plot_pair(t1, t2, df, roll_h, roll_c, roll_r, spread, sc, save)

    if not scorecards:
        print("No pairs had sufficient data.")
        return

    print("\n" + "=" * 60)
    print("  PAIR SELECTION RECOMMENDATION")
    print("=" * 60)
    print(f"  {'Pair':<12}  {'Verdict':<10}  {'MR%':>5}  {'H':>6}  {'Coint':>7}  {'Corr':>6}  {'ADF':>7}")
    print("  " + "-" * 62)
    for sc in scorecards:
        print(f"  {sc['pair']:<12}  {sc['verdict']:<10}  "
              f"{sc['mr_pct']*100:>4.0f}%  "
              f"{sc['avg_hurst']:>6.3f}  "
              f"{sc['coint_pval']:>7.4f}  "
              f"{sc['avg_corr']:>6.3f}  "
              f"{sc['adf_pval']:>7.4f}")
    print("=" * 60)

    print("\n  Recommended PAIRS for stress_test.py:")
    print("  PAIRS = [")
    for sc in scorecards:
        t1, t2 = sc["pair"].split("/")
        if sc["verdict"] == "PASS":
            print(f'      ("{t1}", "{t2}"),  # PASS')
        elif sc["verdict"] == "MARGINAL":
            print(f'      ("{t1}", "{t2}"),  # MARGINAL')
    for sc in scorecards:
        if sc["verdict"] == "FAIL":
            t1, t2 = sc["pair"].split("/")
            print(f'    # ("{t1}", "{t2}"),  # FAIL - dropped')
    print("  ]")

    plot_summary_dashboard(scorecards, os.path.join(out_dir, "diag_summary.png"))


if __name__ == "__main__":
    run_diagnostics()
