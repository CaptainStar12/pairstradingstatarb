"""
walk_forward.py
===============
Walk-forward out-of-sample test for the NSAIEngine on JPM/BAC.

Methodology
-----------
1.  Download the maximum available 5-minute history (60 days via yfinance).
2.  Split into rolling windows:
      - TRAIN window  : engine warms up, Kalman converges  (not traded)
      - TEST  window  : signals are live-traded             (out-of-sample)
3.  Each fold gets a FRESH engine instance — no parameter leakage between folds.
4.  Results across all folds are aggregated and plotted.

Key question being tested
--------------------------
Does the JPM/BAC edge appear consistently across multiple non-overlapping
test windows, or is it concentrated in one lucky stretch?

If Sharpe > 1 and positive return in 3+ of 4 folds → structurally interesting.
If only 1-2 folds positive              → regime-specific / coincidence.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from nsaibrain import NSAIEngine
import os

# ── Config ────────────────────────────────────────────────────────────────────
T1             = "JPM"
T2             = "BAC"
Z_THRESHOLD    = 1.2
SHARE_SIZE     = 20          # realistic: 20 sh x ~$220 = ~$4,400 exposure
STARTING       = 5000.0
EXEC_DELAY     = 1
ROUND_TRIP_FEE = 10.0        # $1 commission + $5 slippage + $4 spread
WARMUP_BARS    = 100         # bars used to warm up Kalman before trading

# Walk-forward window sizes (in bars, ~78 bars per trading day)
TRAIN_BARS     = 390         # ~5 trading days of warmup per fold
TEST_BARS      = 390         # ~5 trading days of out-of-sample trading

SAVE_DIR       = os.path.dirname(os.path.abspath(__file__))


# ── Core engine loop (operates on a pre-loaded DataFrame slice) ───────────────
def run_fold(df_slice, fold_label=""):
    """
    Run one walk-forward fold on df_slice.
    First TRAIN_BARS rows are warmup only (engine runs, no trades).
    Remaining rows are live-traded out-of-sample.
    Returns equity curve, trade list, z-series.
    """
    engine        = NSAIEngine(share_size=SHARE_SIZE, round_trip_fee=ROUND_TRIP_FEE)
    cash          = STARTING
    pos           = 0
    entry_spread  = 0.0
    entry_i       = 0
    equity_curve  = []
    trades        = []
    z_series      = []
    pending_signal = None
    pending_bar    = -1

    for i in range(len(df_slice)):
        p1 = float(df_slice["P1"].iloc[i])
        p2 = float(df_slice["P2"].iloc[i])

        # Execute pending signal if delay elapsed
        if pending_signal is not None and i >= pending_bar + EXEC_DELAY:
            paction, _ = pending_signal
            fill_spread = p1 - (engine.x[0] * p2)

            if paction in ("LONG", "SHORT") and pos == 0:
                pos          = 1 if paction == "LONG" else -1
                entry_spread = fill_spread
                entry_i      = i
                cash        -= engine.round_trip_fee / 2

            elif paction in ("EXIT", "EXIT_TIMEOUT") and pos != 0:
                pnl   = pos * (fill_spread - entry_spread) * engine.share_size
                cash += pnl - (engine.round_trip_fee / 2)
                trades.append({
                    "fold":         fold_label,
                    "entry_bar":    entry_i,
                    "exit_bar":     i,
                    "direction":    "LONG" if pos == 1 else "SHORT",
                    "entry_spread": entry_spread,
                    "exit_spread":  fill_spread,
                    "gross_pnl":    pnl,
                    "pnl":          pnl - engine.round_trip_fee,
                    "exit_reason":  paction,
                    "bars_held":    i - entry_i,
                })
                pos          = 0
                entry_spread = 0.0

            pending_signal = None

        # Generate signal
        res    = engine.get_signal(p1, p2, current_pos=pos,
                                   entry_spread=entry_spread,
                                   z_threshold=Z_THRESHOLD)
        action = res["action"]
        z      = res["z"]
        z_series.append(z)

        # Only trade after warmup (TRAIN_BARS rows are observation-only)
        in_warmup = (i < TRAIN_BARS)
        equity_curve.append(cash)

        if in_warmup:
            continue

        if pending_signal is None:
            if pos == 0 and action in ("LONG", "SHORT"):
                pending_signal = (action, res["spread"])
                pending_bar    = i
            elif pos != 0 and action in ("EXIT", "EXIT_TIMEOUT"):
                pending_signal = (action, res["spread"])
                pending_bar    = i

    # Force-close
    if pos != 0:
        last_spread = float(df_slice["P1"].iloc[-1]) - (
            engine.x[0] * float(df_slice["P2"].iloc[-1]))
        final_pnl = pos * (last_spread - entry_spread) * engine.share_size
        cash += final_pnl - (engine.round_trip_fee / 2)
        trades.append({
            "fold":         fold_label,
            "entry_bar":    entry_i,
            "exit_bar":     len(df_slice) - 1,
            "direction":    "LONG" if pos == 1 else "SHORT",
            "entry_spread": entry_spread,
            "exit_spread":  last_spread,
            "gross_pnl":    final_pnl,
            "pnl":          final_pnl - engine.round_trip_fee,
            "exit_reason":  "FORCE_CLOSE",
            "bars_held":    len(df_slice) - 1 - entry_i,
        })
        equity_curve[-1] = cash

    return {
        "label":         fold_label,
        "equity":        equity_curve,
        "trades":        pd.DataFrame(trades),
        "z_series":      z_series,
        "final_cash":    cash,
        "starting_cash": STARTING,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────
def fold_stats(result):
    eq       = np.array(result["equity"])
    trades   = result["trades"]
    final    = result["final_cash"]
    ret      = (final - STARTING) / STARTING * 100

    peak     = np.maximum.accumulate(eq)
    dd       = (eq - peak) / peak * 100
    max_dd   = dd.min()

    returns  = np.diff(eq)
    sharpe   = (returns.mean() / returns.std() * np.sqrt(252 * 78)
                if returns.std() > 0 else 0)

    n        = len(trades)
    win_rate = (trades["pnl"] > 0).mean() * 100 if n > 0 else 0
    avg_win  = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
    avg_loss = trades.loc[trades["pnl"] <= 0, "pnl"].mean() if (trades["pnl"] <= 0).any() else 0
    gw       = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gl       = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf       = gw / gl if gl > 0 else float("inf")
    avg_bars = trades["bars_held"].mean() if n > 0 else 0

    return {
        "Return (%)":    ret,
        "Sharpe":        sharpe,
        "Max DD (%)":    max_dd,
        "Trades":        n,
        "Win Rate (%)":  win_rate,
        "Avg Win ($)":   avg_win,
        "Avg Loss ($)":  avg_loss,
        "Profit Factor": pf,
        "Avg Bars":      avg_bars,
        "drawdown":      dd,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_walk_forward(folds, stats_list, all_trades):
    n_folds = len(folds)

    GOLD  = "#c9a84c"; GREEN = "#3ddc97"; RED   = "#ff5f6d"
    BLUE  = "#4fa3e0"; DIM   = "#888899"; BG    = "#0d0d0f"; PANEL = "#14141a"
    AMBER = "#ffb347"

    def style(ax):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a35")
        ax.yaxis.label.set_color(DIM)
        ax.xaxis.label.set_color(DIM)
        ax.title.set_color(GOLD)

    dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")
    pct_fmt    = FuncFormatter(lambda x, _: f"{x:.1f}%")

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.text(0.5, 0.97, f"NSAI ENGINE  \u00b7  {T1}/{T2}  WALK-FORWARD TEST",
             ha="center", fontsize=14, color=GOLD,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.955,
             f"Train={TRAIN_BARS} bars  |  Test={TEST_BARS} bars  |  "
             f"Z={Z_THRESHOLD}  |  Shares={SHARE_SIZE}  |  "
             f"Fee=${ROUND_TRIP_FEE:.0f}RT  |  Delay={EXEC_DELAY}bar  |  "
             f"{n_folds} folds",
             ha="center", fontsize=8.5, color=DIM, fontfamily="monospace")

    # Layout: top row = stitched equity + drawdown
    #         middle  = per-fold equity mini-charts
    #         bottom  = aggregate PnL bars + summary table
    gs = gridspec.GridSpec(
        3, n_folds,
        figure=fig, hspace=0.5, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.06
    )

    fold_colors = [GREEN, BLUE, AMBER, "#c77dff", "#ff9f1c", "#2ec4b6"]

    # ── Row 0: per-fold equity curves ────────────────────────────────────────
    for fi, (fold, stats) in enumerate(zip(folds, stats_list)):
        ax = fig.add_subplot(gs[0, fi])
        style(ax)
        eq   = fold["equity"]
        col  = fold_colors[fi % len(fold_colors)]
        base = STARTING

        ax.plot(eq, color=col, lw=1.1, zorder=3)
        ax.fill_between(range(len(eq)), eq, base,
                        where=np.array(eq) >= base, color=GREEN, alpha=0.1)
        ax.fill_between(range(len(eq)), eq, base,
                        where=np.array(eq) <  base, color=RED,   alpha=0.12)
        ax.axhline(base, color=DIM, lw=0.6, ls="--")
        ax.axvline(TRAIN_BARS, color=AMBER, lw=0.8, ls=":",
                   label="train|test")

        # Mark trades
        if len(fold["trades"]):
            for _, tr in fold["trades"].iterrows():
                ax.axvline(tr["entry_bar"],
                           color=GREEN if tr["pnl"] > 0 else RED,
                           alpha=0.2, lw=0.5)

        ax.yaxis.set_major_formatter(dollar_fmt)
        ret_str = f"{stats['Return (%)']:+.1f}%"
        ret_col = GREEN if stats["Return (%)"] > 0 else RED
        ax.set_title(f"Fold {fi+1}  ({ret_str})", fontsize=8,
                     fontfamily="monospace", color=ret_col, loc="left")
        ax.set_xlabel("Bar", fontsize=7)

        # Annotate warmup zone
        ax.axvspan(0, TRAIN_BARS, color=AMBER, alpha=0.04)
        ax.text(TRAIN_BARS / 2, min(eq) + (max(eq) - min(eq)) * 0.05,
                "TRAIN", ha="center", fontsize=6,
                color=AMBER, fontfamily="monospace", alpha=0.7)

    # ── Row 1: per-fold PnL bars + Z-score ────────────────────────────────────
    for fi, (fold, stats) in enumerate(zip(folds, stats_list)):
        ax = fig.add_subplot(gs[1, fi])
        style(ax)
        trades = fold["trades"]

        if len(trades):
            # Only show test-window trades (entry_bar >= TRAIN_BARS)
            test_trades = trades[trades["entry_bar"] >= TRAIN_BARS]
            if len(test_trades):
                cols = [GREEN if p > 0 else RED for p in test_trades["pnl"]]
                ax.bar(range(len(test_trades)), test_trades["pnl"],
                       color=cols, width=0.7)
                ax.axhline(0, color=DIM, lw=0.5)
                ax.yaxis.set_major_formatter(dollar_fmt)

        n    = stats["Trades"]
        wr   = stats["Win Rate (%)"]
        pf   = stats["Profit Factor"]
        sh   = stats["Sharpe"]
        ax.set_title(
            f"n={n}  WR={wr:.0f}%  PF={pf:.1f}  Sh={sh:.2f}",
            fontsize=7, fontfamily="monospace", color=DIM, loc="left"
        )
        ax.set_xlabel("Trade #", fontsize=7)

    # ── Row 2 left half: stitched equity across all folds ────────────────────
    ax_eq = fig.add_subplot(gs[2, : n_folds // 2 + (n_folds % 2)])
    style(ax_eq)

    offset = 0
    stitch_eq = []
    stitch_cash = STARTING
    for fi, fold in enumerate(folds):
        test_eq = fold["equity"][TRAIN_BARS:]
        # Normalize each fold to continue from previous fold's end cash
        delta = np.array(test_eq) - STARTING
        seg   = stitch_cash + delta
        stitch_eq.extend(seg.tolist())
        stitch_cash = seg[-1]

    ax_eq.plot(stitch_eq, color=BLUE, lw=1.1, zorder=3)
    ax_eq.fill_between(range(len(stitch_eq)), stitch_eq, STARTING,
                       where=np.array(stitch_eq) >= STARTING,
                       color=GREEN, alpha=0.12)
    ax_eq.fill_between(range(len(stitch_eq)), stitch_eq, STARTING,
                       where=np.array(stitch_eq) <  STARTING,
                       color=RED, alpha=0.15)
    ax_eq.axhline(STARTING, color=DIM, lw=0.6, ls="--")

    # Fold boundaries on stitched chart
    boundary = 0
    for fi, fold in enumerate(folds):
        boundary += len(fold["equity"][TRAIN_BARS:])
        ax_eq.axvline(boundary, color=AMBER, lw=0.7,
                      ls="--", alpha=0.5,
                      label=f"Fold {fi+1}" if fi == 0 else "")
        ax_eq.text(boundary - len(fold["equity"][TRAIN_BARS:]) / 2,
                   min(stitch_eq) + (max(stitch_eq) - min(stitch_eq)) * 0.03,
                   f"F{fi+1}", ha="center", fontsize=7,
                   color=AMBER, fontfamily="monospace")

    ax_eq.yaxis.set_major_formatter(dollar_fmt)
    final_ret = (stitch_cash - STARTING) / STARTING * 100
    ax_eq.set_title(
        f"STITCHED OUT-OF-SAMPLE EQUITY  ({final_ret:+.1f}% cumulative)",
        fontsize=9, fontfamily="monospace", loc="left"
    )

    # ── Row 2 right half: summary stats table ────────────────────────────────
    ax_st = fig.add_subplot(gs[2, n_folds // 2 + (n_folds % 2):])
    style(ax_st)
    ax_st.axis("off")
    ax_st.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_st.transAxes,
                                   fill=True, facecolor=PANEL, zorder=0))

    headers = ["Fold", "Return", "Sharpe", "Trades", "WinRate", "PF", "MaxDD"]
    col_x   = [0.01, 0.16, 0.31, 0.45, 0.57, 0.72, 0.85]
    y_start = 0.93
    row_h   = 0.11

    # Header row
    for hdr, cx in zip(headers, col_x):
        ax_st.text(cx, y_start, hdr, transform=ax_st.transAxes,
                   fontsize=7.5, color=GOLD, fontfamily="monospace",
                   va="top", fontweight="bold")

    ax_st.plot([0.01, 0.99], [y_start - 0.005, y_start - 0.005],
               color=DIM, lw=0.5, transform=ax_st.transAxes)

    # Fold rows
    for fi, stats in enumerate(stats_list):
        y = y_start - (fi + 1) * row_h
        ret_c = GREEN if stats["Return (%)"] > 0 else RED
        sh_c  = GREEN if stats["Sharpe"] > 1 else (AMBER if stats["Sharpe"] > 0 else RED)
        pf_c  = GREEN if stats["Profit Factor"] > 1 else RED
        wr_c  = GREEN if stats["Win Rate (%)"] > 50 else RED

        vals = [
            (f"Fold {fi+1}",                          GOLD),
            (f"{stats['Return (%)']:+.1f}%",           ret_c),
            (f"{stats['Sharpe']:.2f}",                 sh_c),
            (f"{stats['Trades']}",                     DIM),
            (f"{stats['Win Rate (%)']:.0f}%",          wr_c),
            (f"{stats['Profit Factor']:.2f}",          pf_c),
            (f"{stats['Max DD (%)']:.1f}%",            RED),
        ]
        for (val, col), cx in zip(vals, col_x):
            ax_st.text(cx, y, val, transform=ax_st.transAxes,
                       fontsize=7.5, color=col,
                       fontfamily="monospace", va="top")

    # Aggregate row
    total_trades = sum(s["Trades"] for s in stats_list)
    all_pnl      = all_trades["pnl"] if len(all_trades) > 0 else pd.Series(dtype=float)
    agg_wr       = (all_pnl > 0).mean() * 100 if len(all_pnl) > 0 else 0
    agg_gw       = all_pnl[all_pnl > 0].sum()
    agg_gl       = abs(all_pnl[all_pnl < 0].sum())
    agg_pf       = agg_gw / agg_gl if agg_gl > 0 else float("inf")
    agg_ret      = (stitch_cash - STARTING) / STARTING * 100
    agg_sh       = np.mean([s["Sharpe"] for s in stats_list])

    y = y_start - (len(stats_list) + 1) * row_h
    ax_st.plot([0.01, 0.99], [y + row_h * 0.85, y + row_h * 0.85],
               color=DIM, lw=0.4, transform=ax_st.transAxes)

    agg_vals = [
        ("AGGREGATE",              GOLD),
        (f"{agg_ret:+.1f}%",       GREEN if agg_ret > 0 else RED),
        (f"{agg_sh:.2f}",          GREEN if agg_sh > 1 else RED),
        (f"{total_trades}",        DIM),
        (f"{agg_wr:.0f}%",         GREEN if agg_wr > 50 else RED),
        (f"{agg_pf:.2f}",          GREEN if agg_pf > 1 else RED),
        ("--",                     DIM),
    ]
    for (val, col), cx in zip(agg_vals, col_x):
        ax_st.text(cx, y, val, transform=ax_st.transAxes,
                   fontsize=7.5, color=col,
                   fontfamily="monospace", va="top", fontweight="bold")

    ax_st.set_title("FOLD-BY-FOLD SUMMARY", fontsize=9,
                    fontfamily="monospace", loc="left")

    save_path = os.path.join(SAVE_DIR, "nsai_walk_forward.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"[saved] {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_walk_forward():
    print(f"[+] Downloading {T1}/{T2} 5-minute history...")
    d1 = yf.download(T1, period="60d", interval="5m",
                     auto_adjust=True, progress=False)
    d2 = yf.download(T2, period="60d", interval="5m",
                     auto_adjust=True, progress=False)
    df = pd.merge(d1["Close"], d2["Close"],
                  left_index=True, right_index=True).dropna()
    df.columns = ["P1", "P2"]
    print(f"[+] {len(df)} bars loaded.")

    fold_size = TRAIN_BARS + TEST_BARS
    n_folds   = len(df) // TEST_BARS - 1  # rolling, non-overlapping test windows
    n_folds   = max(2, min(n_folds, 6))   # cap at 6 folds for readability

    print(f"[+] Running {n_folds} walk-forward folds "
          f"(train={TRAIN_BARS} bars, test={TEST_BARS} bars each)...")

    folds      = []
    stats_list = []

    for fi in range(n_folds):
        # Each fold: start at fi * TEST_BARS, length = TRAIN + TEST
        start = fi * TEST_BARS
        end   = start + fold_size
        if end > len(df):
            break

        slice_df = df.iloc[start:end].reset_index(drop=True)
        label    = f"Fold {fi+1}  [bars {start}-{end}]"
        result   = run_fold(slice_df, fold_label=f"F{fi+1}")
        stats    = fold_stats(result)
        folds.append(result)
        stats_list.append(stats)

        pos_str = "+" if stats["Return (%)"] > 0 else ""
        print(f"  Fold {fi+1}: {stats['Trades']:>3} trades | "
              f"Return {pos_str}{stats['Return (%)']:.2f}% | "
              f"Sharpe {stats['Sharpe']:.2f} | "
              f"WinRate {stats['Win Rate (%)']:.0f}% | "
              f"PF {stats['Profit Factor']:.2f}")

    if not folds:
        print("Not enough data for walk-forward test.")
        return

    # Aggregate all test-window trades
    all_trades = pd.concat(
        [f["trades"][f["trades"]["entry_bar"] >= TRAIN_BARS]
         for f in folds if len(f["trades"]) > 0],
        ignore_index=True
    ) if any(len(f["trades"]) > 0 for f in folds) else pd.DataFrame()

    # Console summary
    print("\n" + "=" * 60)
    print("  WALK-FORWARD VERDICT")
    print("=" * 60)
    positive_folds = sum(1 for s in stats_list if s["Return (%)"] > 0)
    sharpe_above_1 = sum(1 for s in stats_list if s["Sharpe"] > 1.0)
    print(f"  Positive return folds : {positive_folds} / {len(folds)}")
    print(f"  Sharpe > 1.0 folds    : {sharpe_above_1} / {len(folds)}")

    if positive_folds >= len(folds) * 0.75:
        verdict = "STRUCTURALLY INTERESTING — edge appears across multiple windows"
    elif positive_folds >= len(folds) * 0.5:
        verdict = "MIXED — edge present but inconsistent, further research needed"
    else:
        verdict = "REGIME-SPECIFIC — edge concentrated in one period, likely coincidence"
    print(f"\n  Verdict: {verdict}")
    print("=" * 60)

    plot_walk_forward(folds, stats_list, all_trades)


if __name__ == "__main__":
    run_walk_forward()
