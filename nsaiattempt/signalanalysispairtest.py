"""
signal_analysis.py
==================
Asks one narrow, rigorous question:

    Does the Kalman innovation Z-score contain genuine directional
    signal about next-bar spread change in JPM/BAC?

This is independent of trading rules, execution, fees, or initialization.
It tests the ENGINE'S CORE SIGNAL, not a backtest.

Tests performed
---------------
1.  PREDICTIVE REGRESSION
    next_spread_change ~ beta * z_score + epsilon
    If beta is negative and significant → high Z predicts spread FALLS
    (mean-reversion confirmed). t-stat and p-value reported.

2.  QUINTILE ANALYSIS
    Sort all bars by Z-score into 5 buckets.
    Measure average next-bar spread change in each bucket.
    Mean-reversion signal: bucket 5 (highest Z) should have most negative
    forward return; bucket 1 (lowest Z) should have most positive.

3.  SIGN ACCURACY
    When Z > threshold, does spread fall next bar? (SHORT accuracy)
    When Z < -threshold, does spread rise next bar? (LONG accuracy)
    Reported at multiple thresholds.

4.  AUTOCORRELATION OF SPREAD
    Confirms whether the spread itself is mean-reverting
    (necessary condition for the strategy to work at all).

5.  HURST EXPONENT DISTRIBUTION
    Shows what fraction of bars the Hurst filter would actually pass,
    and whether those bars have better signal than the rest.

All tests use the full 60-day history with a single continuous engine
run (no restarts) — this is the condition under which the signal exists.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy import stats as scipy_stats
from nsaibrain import NSAIEngine
import os

# ── Config ────────────────────────────────────────────────────────────────────
T1          = "JPM"
T2          = "BAC"
WARMUP_BARS = 100
Z_THRESH_LIST = [0.5, 1.0, 1.2, 1.5, 2.0]
SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))


# ── Step 1: Build full signal series ─────────────────────────────────────────
def build_signal_series(df):
    """
    Run the engine continuously over all bars.
    Collect z-score, spread, hurst at every bar.
    No trading — pure signal extraction.
    """
    engine = NSAIEngine(share_size=1, round_trip_fee=0)
    records = []

    for i in range(len(df)):
        p1 = float(df["P1"].iloc[i])
        p2 = float(df["P2"].iloc[i])
        res = engine.get_signal(p1, p2, current_pos=0, z_threshold=99)

        records.append({
            "bar":    i,
            "p1":     p1,
            "p2":     p2,
            "z":      res["z"],
            "spread": res["spread"],
            "hurst":  res.get("hurst", 0.5),
            "beta":   engine.x[0],
            "alpha":  engine.x[1],
        })

    sig = pd.DataFrame(records)

    # Forward spread change: how much does spread move on the NEXT bar?
    sig["next_spread"] = sig["spread"].shift(-1)
    sig["fwd_change"]  = sig["next_spread"] - sig["spread"]

    # Drop warmup and last bar (no forward return)
    sig = sig.iloc[WARMUP_BARS:-1].copy()
    sig = sig.dropna(subset=["fwd_change", "z"])
    return sig


# ── Test 1: Predictive Regression ────────────────────────────────────────────
def test_regression(sig):
    x = sig["z"].values
    y = sig["fwd_change"].values

    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    t_stat = slope / se
    n      = len(x)
    r2     = r ** 2

    print("\n── TEST 1: PREDICTIVE REGRESSION ──────────────────────────")
    print(f"   next_spread_change = {slope:.6f} * Z + {intercept:.6f}")
    print(f"   t-statistic : {t_stat:.3f}")
    print(f"   p-value     : {p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else '(not significant)'}")
    print(f"   R²          : {r2:.5f}")
    print(f"   N           : {n}")
    if slope < 0 and p < 0.05:
        print("   >> Negative slope + significant p: Z predicts mean-REVERSION ✓")
    elif slope > 0 and p < 0.05:
        print("   >> Positive slope + significant p: Z predicts mean-EXPANSION ✗")
    else:
        print("   >> No significant predictive relationship found")

    return {"slope": slope, "intercept": intercept, "t": t_stat, "p": p, "r2": r2, "n": n}


# ── Test 2: Quintile Analysis ─────────────────────────────────────────────────
def test_quintiles(sig):
    sig = sig.copy()
    sig["quintile"] = pd.qcut(sig["z"], 5, labels=["Q1\n(low Z)", "Q2", "Q3", "Q4", "Q5\n(high Z)"])
    qt = sig.groupby("quintile", observed=True)["fwd_change"].agg(["mean", "std", "count"])
    qt["stderr"] = qt["std"] / np.sqrt(qt["count"])
    qt["t_stat"] = qt["mean"] / qt["stderr"]

    print("\n── TEST 2: QUINTILE ANALYSIS ───────────────────────────────")
    print(f"   {'Quintile':<12} {'Avg Fwd Chg':>12} {'t-stat':>8} {'N':>6}")
    print("   " + "-" * 42)
    for q, row in qt.iterrows():
        sig_marker = "*" if abs(row["t_stat"]) > 1.96 else " "
        print(f"   {str(q):<12} {row['mean']:>+12.5f} {row['t_stat']:>7.2f}{sig_marker} {int(row['count']):>6}")

    monotone = all(qt["mean"].iloc[i] >= qt["mean"].iloc[i+1]
                   for i in range(len(qt)-1))
    print(f"\n   Monotone decreasing (Q1 high → Q5 low): {monotone}")
    if monotone:
        print("   >> Perfect quintile ordering confirms mean-reversion signal ✓")

    return qt


# ── Test 3: Sign Accuracy ─────────────────────────────────────────────────────
def test_sign_accuracy(sig):
    print("\n── TEST 3: DIRECTIONAL ACCURACY AT THRESHOLDS ─────────────")
    print(f"   {'Threshold':>10} {'SHORT acc':>12} {'LONG acc':>12} "
          f"{'N signals':>10} {'Coverage':>10}")
    print("   " + "-" * 58)

    results = []
    total = len(sig)
    for thresh in Z_THRESH_LIST:
        short_mask = sig["z"] >  thresh
        long_mask  = sig["z"] < -thresh

        # SHORT accurate if spread falls next bar
        short_acc = (sig.loc[short_mask, "fwd_change"] < 0).mean() if short_mask.sum() > 0 else np.nan
        long_acc  = (sig.loc[long_mask,  "fwd_change"] > 0).mean() if long_mask.sum()  > 0 else np.nan
        n_sigs    = short_mask.sum() + long_mask.sum()
        coverage  = n_sigs / total * 100

        combined  = np.nanmean([short_acc, long_acc])
        sig_mark  = "✓" if combined > 0.55 else ("~" if combined > 0.50 else "✗")

        print(f"   {thresh:>10.1f} {short_acc:>11.1%} {long_acc:>12.1%} "
              f"{n_sigs:>10} {coverage:>9.1f}%  {sig_mark}")
        results.append({
            "threshold":  thresh,
            "short_acc":  short_acc,
            "long_acc":   long_acc,
            "n_signals":  n_sigs,
            "coverage":   coverage,
            "combined":   combined,
        })
    return pd.DataFrame(results)


# ── Test 4: Spread Autocorrelation ───────────────────────────────────────────
def test_autocorrelation(sig):
    spread = sig["spread"].values
    lags   = [1, 2, 3, 5, 10, 20]

    print("\n── TEST 4: SPREAD AUTOCORRELATION ─────────────────────────")
    print(f"   {'Lag (bars)':>12} {'AutoCorr':>10} {'p-value':>10}")
    print("   " + "-" * 36)

    acorrs = []
    for lag in lags:
        r, p = scipy_stats.pearsonr(spread[:-lag], spread[lag:])
        sig_mark = "*" if p < 0.05 else " "
        print(f"   {lag:>12} {r:>+10.4f} {p:>9.4f}{sig_mark}")
        acorrs.append({"lag": lag, "r": r, "p": p})

    # First-difference autocorrelation (mean-reversion test)
    diff = np.diff(spread)
    r1, p1 = scipy_stats.pearsonr(diff[:-1], diff[1:])
    print(f"\n   First-difference lag-1 autocorr: {r1:+.4f}  (p={p1:.4f})")
    if r1 < 0 and p1 < 0.05:
        print("   >> Negative autocorrelation in changes: spread is mean-reverting ✓")
    else:
        print("   >> No mean-reversion in spread changes detected")

    return pd.DataFrame(acorrs)


# ── Test 5: Hurst Filter Quality ─────────────────────────────────────────────
def test_hurst_filter(sig):
    print("\n── TEST 5: HURST FILTER QUALITY ────────────────────────────")

    mr_mask   = sig["hurst"] < 0.5    # mean-reverting
    other     = sig[~mr_mask]
    mr        = sig[mr_mask]

    print(f"   Bars passing H<0.5 filter : {mr_mask.sum()} / {len(sig)} "
          f"({mr_mask.mean()*100:.1f}%)")

    if len(mr) > 10 and len(other) > 10:
        # Does Z predict better when Hurst says mean-reverting?
        slope_mr,  _, _, p_mr,  _ = scipy_stats.linregress(mr["z"],    mr["fwd_change"])
        slope_all, _, _, p_all, _ = scipy_stats.linregress(sig["z"],   sig["fwd_change"])

        print(f"\n   Regression slope (all bars)       : {slope_all:.6f}  p={p_all:.4f}")
        print(f"   Regression slope (H<0.5 only)     : {slope_mr:.6f}  p={p_mr:.4f}")

        if abs(slope_mr) > abs(slope_all):
            print("   >> Hurst filter IMPROVES signal quality ✓")
        else:
            print("   >> Hurst filter does NOT improve signal quality")

    for thresh in [0.4, 0.45, 0.5]:
        mask = sig["hurst"] < thresh
        pct  = mask.mean() * 100
        if mask.sum() > 5:
            slope, _, _, p, _ = scipy_stats.linregress(
                sig.loc[mask, "z"], sig.loc[mask, "fwd_change"])
            print(f"   H<{thresh}: {pct:4.1f}% of bars | slope={slope:.6f} p={p:.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_signal_analysis(sig, reg, qt, acc, acorr):
    GOLD  = "#c9a84c"; GREEN = "#3ddc97"; RED   = "#ff5f6d"
    BLUE  = "#4fa3e0"; DIM   = "#888899"; BG    = "#0d0d0f"; PANEL = "#14141a"

    def style(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a35")
        ax.yaxis.label.set_color(DIM)
        ax.xaxis.label.set_color(DIM)
        ax.title.set_color(GOLD)

    fig = plt.figure(figsize=(16, 13), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                            left=0.07, right=0.97, top=0.91, bottom=0.06)

    fig.text(0.5, 0.96,
             f"NSAI ENGINE  \u00b7  {T1}/{T2}  SIGNAL ANALYSIS",
             ha="center", fontsize=14, color=GOLD,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.945,
             "Does the Kalman Innovation Z-score predict next-bar spread reversion?",
             ha="center", fontsize=9, color=DIM, fontfamily="monospace")

    # ── 1. Scatter: Z vs next-bar spread change ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
    sample = sig.sample(min(2000, len(sig)), random_state=42)
    ax1.scatter(sample["z"], sample["fwd_change"],
                alpha=0.15, s=4, color=BLUE, rasterized=True)

    # Regression line
    xr = np.linspace(sig["z"].quantile(0.01), sig["z"].quantile(0.99), 100)
    yr = reg["slope"] * xr + reg["intercept"]
    lc = GREEN if reg["slope"] < 0 else RED
    ax1.plot(xr, yr, color=lc, lw=1.5, label=f"slope={reg['slope']:.5f}")
    ax1.axhline(0, color=DIM, lw=0.5)
    ax1.axvline(0, color=DIM, lw=0.5)
    ax1.set_xlabel("Z-score (current bar)", fontsize=8)
    ax1.set_ylabel("Spread change (next bar)", fontsize=8)
    p_str = f"p={reg['p']:.4f}" + (" ***" if reg["p"] < 0.05 else "")
    ax1.set_title(f"Z → Next Spread  {p_str}", fontsize=9,
                  fontfamily="monospace", loc="left")
    ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM)

    # ── 2. Quintile bar chart ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    q_labels = [str(q) for q in qt.index]
    q_means  = qt["mean"].values
    q_errs   = qt["stderr"].values
    q_colors = [GREEN if v > 0 else RED for v in q_means]
    bars = ax2.bar(q_labels, q_means, color=q_colors,
                   yerr=q_errs, capsize=3,
                   error_kw={"ecolor": DIM, "lw": 0.8})
    ax2.axhline(0, color=DIM, lw=0.7)
    ax2.set_xlabel("Z-score quintile", fontsize=8)
    ax2.set_ylabel("Avg next-bar spread change", fontsize=8)
    ax2.set_title("Quintile Forward Returns", fontsize=9,
                  fontfamily="monospace", loc="left")

    # ── 3. Sign accuracy by threshold ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2]); style(ax3)
    thresholds = acc["threshold"].values
    short_accs = acc["short_acc"].values * 100
    long_accs  = acc["long_acc"].values * 100
    x = np.arange(len(thresholds))
    w = 0.35
    ax3.bar(x - w/2, short_accs, width=w, color=RED,   alpha=0.8, label="SHORT acc")
    ax3.bar(x + w/2, long_accs,  width=w, color=GREEN, alpha=0.8, label="LONG acc")
    ax3.axhline(50, color=DIM, lw=0.8, ls="--", label="50% (random)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Z>{t}" for t in thresholds], fontsize=7)
    ax3.set_ylabel("Next-bar accuracy (%)", fontsize=8)
    ax3.set_title("Directional Accuracy", fontsize=9,
                  fontfamily="monospace", loc="left")
    ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM)
    ax3.set_ylim(30, 75)

    # ── 4. Spread time series ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :]); style(ax4)
    ax4.plot(sig["bar"].values, sig["spread"].values,
             color=BLUE, lw=0.6, alpha=0.9)
    ax4.axhline(sig["spread"].mean(), color=DIM, lw=0.6, ls="--",
                label="mean")

    # Shade regions where |Z| > 1.2
    for _, row in sig[sig["z"].abs() > 1.2].iterrows():
        col = RED if row["z"] > 0 else GREEN
        ax4.axvspan(row["bar"] - 0.5, row["bar"] + 0.5,
                    color=col, alpha=0.3)

    ax4.set_xlabel("Bar", fontsize=8)
    ax4.set_ylabel("Spread (P1 - β·P2)", fontsize=8)
    ax4.set_title(
        "Spread Series  (shaded = |Z|>1.2  RED=short signal  GREEN=long signal)",
        fontsize=9, fontfamily="monospace", loc="left")
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM)

    # ── 5. Autocorrelation bar chart ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0]); style(ax5)
    lags = acorr["lag"].values
    rs   = acorr["r"].values
    cols = [GREEN if r < 0 else RED for r in rs]
    ax5.bar(lags, rs, color=cols, width=0.6)
    ax5.axhline(0, color=DIM, lw=0.7)
    ax5.set_xlabel("Lag (bars)", fontsize=8)
    ax5.set_ylabel("Autocorrelation", fontsize=8)
    ax5.set_title("Spread Autocorrelation", fontsize=9,
                  fontfamily="monospace", loc="left")

    # ── 6. Z-score distribution ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1]); style(ax6)
    z_vals = sig["z"].values
    ax6.hist(z_vals, bins=80, color=BLUE, alpha=0.7, density=True)
    xn = np.linspace(z_vals.min(), z_vals.max(), 200)
    ax6.plot(xn, scipy_stats.norm.pdf(xn, z_vals.mean(), z_vals.std()),
             color=GOLD, lw=1.2, label="Normal fit")
    for thresh in [1.2, -1.2]:
        ax6.axvline(thresh, color=RED if thresh > 0 else GREEN,
                    lw=0.9, ls="--")
    ax6.set_xlabel("Z-score", fontsize=8)
    ax6.set_ylabel("Density", fontsize=8)
    ax6.set_title("Z-score Distribution", fontsize=9,
                  fontfamily="monospace", loc="left")
    ax6.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM)

    # ── 7. Summary verdict box ───────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2]); style(ax7); ax7.axis("off")
    ax7.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax7.transAxes,
                                 fill=True, facecolor=PANEL, zorder=0))

    # Score the evidence
    score = 0
    evidence = []

    if reg["slope"] < 0 and reg["p"] < 0.05:
        score += 1
        evidence.append(("Regression slope negative + sig.", GREEN, True))
    else:
        evidence.append(("Regression slope not significant", RED, False))

    best_acc = acc["combined"].max()
    if best_acc > 0.55:
        score += 1
        evidence.append((f"Best directional acc: {best_acc:.1%}", GREEN, True))
    else:
        evidence.append((f"Best directional acc: {best_acc:.1%}", RED, False))

    q_mono = all(qt["mean"].iloc[i] >= qt["mean"].iloc[i+1]
                 for i in range(len(qt)-1))
    if q_mono:
        score += 1
        evidence.append(("Quintiles monotone decreasing", GREEN, True))
    else:
        evidence.append(("Quintiles not monotone", RED, False))

    diff   = np.diff(sig["spread"].values)
    r1, p1 = scipy_stats.pearsonr(diff[:-1], diff[1:])
    if r1 < 0 and p1 < 0.05:
        score += 1
        evidence.append(("Spread mean-reverting (diff AC<0)", GREEN, True))
    else:
        evidence.append(("Spread not mean-reverting", RED, False))

    ax7.set_title("SIGNAL VERDICT", fontsize=9,
                  fontfamily="monospace", loc="left")

    for ei, (txt, col, passed) in enumerate(evidence):
        y = 0.82 - ei * 0.155
        mark = "+" if passed else "-"
        ax7.text(0.04, y, f"[{mark}]", transform=ax7.transAxes,
                 fontsize=9, color=col, fontfamily="monospace", va="top",
                 fontweight="bold")
        ax7.text(0.18, y, txt, transform=ax7.transAxes,
                 fontsize=8, color=DIM, fontfamily="monospace", va="top")

    y_v = 0.82 - len(evidence) * 0.155 - 0.05
    if score >= 3:
        verdict_txt = f"SIGNAL EXISTS ({score}/4)"
        verdict_col = GREEN
    elif score >= 2:
        verdict_txt = f"WEAK SIGNAL ({score}/4)"
        verdict_col = GOLD
    else:
        verdict_txt = f"NO SIGNAL ({score}/4)"
        verdict_col = RED

    ax7.text(0.5, y_v, verdict_txt, transform=ax7.transAxes,
             fontsize=11, color=verdict_col, fontfamily="monospace",
             va="top", ha="center", fontweight="bold")

    save_path = os.path.join(SAVE_DIR, "nsai_signal_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"\n[saved] {save_path}")

    return score, verdict_txt


# ── Main ──────────────────────────────────────────────────────────────────────
def run_signal_analysis():
    print(f"[+] Downloading {T1}/{T2}...")
    d1 = yf.download(T1, period="60d", interval="5m",
                     auto_adjust=True, progress=False)
    d2 = yf.download(T2, period="60d", interval="5m",
                     auto_adjust=True, progress=False)
    df = pd.merge(d1["Close"], d2["Close"],
                  left_index=True, right_index=True).dropna()
    df.columns = ["P1", "P2"]
    print(f"[+] {len(df)} bars. Building signal series (no trading)...")

    sig = build_signal_series(df)
    print(f"[+] {len(sig)} bars after warmup. Running statistical tests...\n")

    reg   = test_regression(sig)
    qt    = test_quintiles(sig)
    acc   = test_sign_accuracy(sig)
    acorr = test_autocorrelation(sig)
    test_hurst_filter(sig)

    print("\n[+] Plotting...")
    score, verdict = plot_signal_analysis(sig, reg, qt, acc, acorr)

    print(f"\n{'='*60}")
    print(f"  FINAL VERDICT: {verdict}")
    print(f"{'='*60}")
    print("""
  Interpretation guide:
  ---------------------
  SIGNAL EXISTS (3-4/4) → The Z-score has genuine predictive
    content. The walk-forward failure is an EXECUTION problem
    (initialization, window size) not a signal problem.
    Worth pursuing with better deployment architecture.

  WEAK SIGNAL (2/4)     → Marginal evidence. Signal may be
    real but too small to survive execution costs.
    Interesting academically, not practically tradeable.

  NO SIGNAL (0-1/4)     → The original backtest result was
    coincidence. The Kalman innovation Z does not predict
    spread reversion in this pair. Stop here.
""")


if __name__ == "__main__":
    run_signal_analysis()
