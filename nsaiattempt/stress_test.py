import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from nsaibrain import NSAIEngine
import os

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nsai_backtest.png")

# ── Best configuration (locked) ───────────────────────────────────────────────
# These parameters produced: +1.82%, Sharpe 1.99, PF 2.20, MaxDD -1.02%
# on the current 5-minute JPM/BAC window with dollar-neutral sizing.
#
# DO NOT change these without running a full validation across multiple windows.
# The goal now is to collect paper trading results using exactly these settings.

Z_THRESHOLD   = 1.0
EXEC_DELAY    = 1
ROUND_TRIP_FEE = 10.0

# ── Position sizing ───────────────────────────────────────────────────────────
# Dollar-neutral: both legs sized to DOLLAR_SIZE notional at entry.
# At $50k/leg the $10 fee is ~0.01% of position — negligible.
SIZING_MODE  = "dollar_neutral"
DOLLAR_SIZE  = 50000.0
FIXED_SHARES = 500        # only used if SIZING_MODE = "fixed"
STARTING     = 120000.0

# ── Stop loss ─────────────────────────────────────────────────────────────────
# 1% of combined notional = $1,000 max loss per trade.
# This is appropriate for mean-reversion — caps catastrophic trades
# without cutting normal reversion-in-progress moves.
STOP_LOSS_PCT = 0.05

# ── Rolling cointegration gate ────────────────────────────────────────────────
# Tests last ROLLING_COINT_BARS bars every bar.
# Only allows new entries when the rolling window is currently cointegrated.
# This ensures we only trade during structurally valid periods.
ROLLING_COINT_GATE   = True
ROLLING_COINT_BARS   = 500
ROLLING_COINT_PVALUE = 0.10

# ── Cointegration screening ───────────────────────────────────────────────────
# The 3-year daily cointegration test currently blocks JPM/BAC (p=0.2426).
# The rolling intraday gate handles regime detection dynamically, so we
# use FORCE_TRADE=True and rely on the rolling gate instead of the static test.
FORCE_TRADE = True

# ── Pairs ─────────────────────────────────────────────────────────────────────
PAIRS = [
    ("JPM", "BAC"),
]

# ── Backtest window ───────────────────────────────────────────────────────────
# Use None/None for current 60-day 5-minute window (production setting).
# For historical hourly testing use START_DATE/END_DATE with INTERVAL="1h".
#
# Yahoo Finance limits:
#   "5m" → last 60 days only (no start/end)
#   "1h" → up to 2 years back (use start/end)

START_DATE  = "2024-11-01"
END_DATE    = "2024-12-31"
INTERVAL    = "1h"
WARMUP_BARS = 30


# ─────────────────────────────────────────────
#  DATA DOWNLOAD
# ─────────────────────────────────────────────
def _fix_close(raw):
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def download_pair_intraday(t1, t2):
    try:
        if START_DATE and END_DATE:
            d1 = yf.download(t1, start=START_DATE, end=END_DATE,
                             interval=INTERVAL, auto_adjust=True, progress=False)
            d2 = yf.download(t2, start=START_DATE, end=END_DATE,
                             interval=INTERVAL, auto_adjust=True, progress=False)
        else:
            d1 = yf.download(t1, period="60d", interval=INTERVAL,
                             auto_adjust=True, progress=False)
            d2 = yf.download(t2, period="60d", interval=INTERVAL,
                             auto_adjust=True, progress=False)
        df = pd.merge(_fix_close(d1).rename("P1"),
                      _fix_close(d2).rename("P2"),
                      left_index=True, right_index=True).dropna()
        return df
    except Exception as e:
        print(f"    download error: {e}")
        return pd.DataFrame()


def download_pair_daily(t1, t2, years=3):
    try:
        d1 = yf.download(t1, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
        d2 = yf.download(t2, period=f"{years}y", interval="1d",
                         auto_adjust=True, progress=False)
        df = pd.merge(_fix_close(d1).rename("P1"),
                      _fix_close(d2).rename("P2"),
                      left_index=True, right_index=True).dropna()
        return df
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  POSITION SIZER
# ─────────────────────────────────────────────
def compute_shares(p1, p2, beta):
    if SIZING_MODE == "fixed":
        return FIXED_SHARES, FIXED_SHARES
    beta_clipped = np.clip(abs(beta), 0.1, 10.0)
    shares_p1    = max(1, int(DOLLAR_SIZE / p1))
    shares_p2    = max(1, int(beta_clipped * DOLLAR_SIZE / p2))
    return shares_p1, shares_p2


# ─────────────────────────────────────────────
#  CORE BACKTEST
# ─────────────────────────────────────────────
def run_backtest(t1, t2, starting_cash=STARTING, z_threshold=Z_THRESHOLD):

    window_str = (f"{START_DATE} → {END_DATE} [{INTERVAL}]"
                  if START_DATE and END_DATE
                  else f"last 60 days [{INTERVAL}]")

    # Download intraday data
    df = download_pair_intraday(t1, t2)
    if len(df) < WARMUP_BARS + 50:
        print(f"  [{t1}/{t2}] insufficient data ({len(df)} bars) — skipped")
        return None

    # Download daily data for OLS initialization
    daily_df = download_pair_daily(t1, t2)

    # OLS initialization from returns-based regression
    if len(daily_df) >= 50:
        ols_beta, ols_alpha = NSAIEngine.ols_beta_alpha(
            daily_df["P1"].values,
            daily_df["P2"].values,
        )
    else:
        ols_beta, ols_alpha = 1.0, 0.0

    print(f"  [{t1}/{t2}] OLS beta={ols_beta:.4f}, alpha={ols_alpha:.4f}")
    print(f"  [{t1}/{t2}] → trading [{window_str}]")

    # Guard: gate window must fit inside available bars.
    # If ROLLING_COINT_BARS >= tradeable bars the buffer never fills,
    # gate_open stays False for the whole run, and result is 0 trades.
    tradeable_bars       = len(df) - WARMUP_BARS
    effective_coint_bars = ROLLING_COINT_BARS
    if ROLLING_COINT_GATE and ROLLING_COINT_BARS >= tradeable_bars:
        effective_coint_bars = max(30, tradeable_bars // 2)
        print(f"  [{t1}/{t2}] WARNING: ROLLING_COINT_BARS ({ROLLING_COINT_BARS}) >= "
              f"tradeable bars ({tradeable_bars}). Auto-clamped to {effective_coint_bars}.")

    # Create and warm up engine
    engine = NSAIEngine(
        round_trip_fee=ROUND_TRIP_FEE,
        require_warmup=True,
        initial_beta=ols_beta,
        initial_alpha=ols_alpha,
    )
    engine.warmup(
        df["P1"].values[:WARMUP_BARS],
        df["P2"].values[:WARMUP_BARS],
    )

    # Trade
    from statsmodels.tsa.stattools import coint as _coint

    cash            = starting_cash
    pos             = 0
    entry_p1        = 0.0
    entry_p2        = 0.0
    entry_shares_p1 = 0
    entry_shares_p2 = 0
    entry_i         = 0
    equity_curve    = []
    trades          = []
    z_series        = []
    pending_signal  = None
    pending_bar     = -1
    p1_buffer       = []
    p2_buffer       = []
    coint_gate_open = not ROLLING_COINT_GATE
    stop_loss_amt   = (DOLLAR_SIZE * 2 * STOP_LOSS_PCT
                       if STOP_LOSS_PCT is not None else None)

    # Diagnostic counters — printed after the loop
    _dbg_gate_open  = 0   # bars where gate was open
    _dbg_hurst_blk  = 0   # gate open but Hurst >= 0.5
    _dbg_z_blk      = 0   # gate+Hurst OK but |Z| < threshold
    _dbg_in_pos     = 0   # bars already in a position

    for i in range(WARMUP_BARS, len(df)):
        p1 = float(df["P1"].iloc[i])
        p2 = float(df["P2"].iloc[i])

        # Execute pending signal
        if pending_signal is not None and i >= pending_bar + EXEC_DELAY:
            paction = pending_signal

            if paction in ("LONG", "SHORT") and pos == 0:
                beta_now = engine.x[0]
                s1, s2   = compute_shares(p1, p2, beta_now)
                pos             = 1 if paction == "LONG" else -1
                entry_p1        = p1
                entry_p2        = p2
                entry_shares_p1 = s1
                entry_shares_p2 = s2
                entry_i         = i
                cash           -= ROUND_TRIP_FEE / 2

            elif paction in ("EXIT", "EXIT_TIMEOUT", "STOP_LOSS") and pos != 0:
                leg1_pnl = pos * entry_shares_p1 * (p1 - entry_p1)
                leg2_pnl = pos * entry_shares_p2 * (p2 - entry_p2)
                pnl      = leg1_pnl - leg2_pnl - ROUND_TRIP_FEE / 2
                cash    += pnl
                trades.append({
                    "pair":        f"{t1}/{t2}",
                    "entry_bar":   entry_i,
                    "exit_bar":    i,
                    "direction":   "LONG" if pos == 1 else "SHORT",
                    "entry_p1":    entry_p1,
                    "entry_p2":    entry_p2,
                    "exit_p1":     p1,
                    "exit_p2":     p2,
                    "shares_p1":   entry_shares_p1,
                    "shares_p2":   entry_shares_p2,
                    "leg1_pnl":    leg1_pnl,
                    "leg2_pnl":    leg2_pnl,
                    "pnl":         pnl - ROUND_TRIP_FEE / 2,
                    "exit_reason": paction,
                    "bars_held":   i - entry_i,
                })
                pos = 0

            pending_signal = None

        # Update rolling buffers
        p1_buffer.append(p1)
        p2_buffer.append(p2)
        if len(p1_buffer) > effective_coint_bars:
            p1_buffer.pop(0)
            p2_buffer.pop(0)

        # Rolling cointegration gate
        if ROLLING_COINT_GATE and len(p1_buffer) >= effective_coint_bars:
            try:
                _, _pval, _ = _coint(
                    np.array(p1_buffer),
                    np.array(p2_buffer),
                )
                coint_gate_open = _pval < ROLLING_COINT_PVALUE
            except Exception:
                coint_gate_open = False

        # Get signal
        res    = engine.get_signal(p1, p2, current_pos=pos,
                                   z_threshold=z_threshold)
        action = res["action"]
        z      = res["z"]
        z_series.append(z)
        equity_curve.append(cash)

        # Diagnostic accounting (only counted after warmup)
        if coint_gate_open:
            _dbg_gate_open += 1
            h_now = engine.get_hurst(engine.spread_history)
            if h_now >= 0.5:
                _dbg_hurst_blk += 1
            elif pos == 0 and abs(z) < z_threshold:
                _dbg_z_blk += 1
            elif pos != 0:
                _dbg_in_pos += 1

        # Stop loss
        if pos != 0 and stop_loss_amt is not None and pending_signal is None:
            leg1 = pos * entry_shares_p1 * (p1 - entry_p1)
            leg2 = pos * entry_shares_p2 * (p2 - entry_p2)
            if (leg1 - leg2) < -stop_loss_amt:
                pending_signal = "STOP_LOSS"
                pending_bar    = i
                continue

        if pending_signal is None:
            if pos == 0 and action in ("LONG", "SHORT") and coint_gate_open:
                pending_signal = action
                pending_bar    = i
            elif pos != 0 and action in ("EXIT", "EXIT_TIMEOUT"):
                pending_signal = action
                pending_bar    = i

    # Diagnostic summary
    _total = len(df) - WARMUP_BARS
    print(f"  [{t1}/{t2}] DIAGNOSTICS ({_total} bars traded):")
    print(f"    Gate open      : {_dbg_gate_open:>5} bars  "
          f"({_dbg_gate_open/_total*100:.1f}%)  "
          f"← 0% means pair not cointegrated at p<{ROLLING_COINT_PVALUE} "
          f"(or gate disabled)")
    print(f"    Hurst blocked  : {_dbg_hurst_blk:>5} bars  "
          f"({_dbg_hurst_blk/max(_dbg_gate_open,1)*100:.1f}% of gate-open) "
          f"← H>=0.5, spread not mean-reverting")
    print(f"    Z too small    : {_dbg_z_blk:>5} bars  "
          f"({_dbg_z_blk/max(_dbg_gate_open,1)*100:.1f}% of gate-open) "
          f"← |Z|<{z_threshold}, no entry signal")
    print(f"    In position    : {_dbg_in_pos:>5} bars  "
          f"({_dbg_in_pos/max(_dbg_gate_open,1)*100:.1f}% of gate-open)")

    # Force-close open position
    if pos != 0:
        p1 = float(df["P1"].iloc[-1])
        p2 = float(df["P2"].iloc[-1])
        leg1_pnl = pos * entry_shares_p1 * (p1 - entry_p1)
        leg2_pnl = pos * entry_shares_p2 * (p2 - entry_p2)
        pnl      = leg1_pnl - leg2_pnl - ROUND_TRIP_FEE / 2
        cash    += pnl
        trades.append({
            "pair":        f"{t1}/{t2}",
            "entry_bar":   entry_i,
            "exit_bar":    len(df) - 1,
            "direction":   "LONG" if pos == 1 else "SHORT",
            "entry_p1":    entry_p1,
            "entry_p2":    entry_p2,
            "exit_p1":     p1,
            "exit_p2":     p2,
            "shares_p1":   entry_shares_p1,
            "shares_p2":   entry_shares_p2,
            "leg1_pnl":    leg1_pnl,
            "leg2_pnl":    leg2_pnl,
            "pnl":         pnl - ROUND_TRIP_FEE / 2,
            "exit_reason": "FORCE_CLOSE",
            "bars_held":   len(df) - 1 - entry_i,
        })
        equity_curve[-1] = cash

    return {
        "equity":       equity_curve,
        "trades":       pd.DataFrame(trades),
        "z_series":     z_series,
        "df":           df,
        "t1": t1, "t2": t2,
        "starting_cash": starting_cash,
        "final_cash":    cash,
        "ols_beta":      ols_beta,
        "ols_alpha":     ols_alpha,
        "window":        window_str,
    }


# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────
def compute_stats(result):
    eq       = np.array(result["equity"])
    trades   = result["trades"]
    starting = result["starting_cash"]
    final    = result["final_cash"]

    if trades.empty or "pnl" not in trades.columns:
        trades = pd.DataFrame(columns=["pnl", "bars_held", "exit_reason"])

    total_return = (final - starting) / starting * 100
    peak         = np.maximum.accumulate(eq)
    drawdown     = (eq - peak) / peak * 100
    max_dd       = drawdown.min()
    returns      = np.diff(eq)
    bpy          = {"5m": 252*78, "1h": 252*7, "1d": 252}.get(INTERVAL, 252*78)
    sharpe       = (returns.mean() / returns.std() * np.sqrt(bpy)) if returns.std() > 0 else 0
    n_trades     = len(trades)
    win_rate     = (trades["pnl"] > 0).mean() * 100 if n_trades > 0 else 0
    gross_win    = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_los    = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    pf           = gross_win / gross_los if gross_los > 0 else float("inf")
    stop_hits    = (trades["exit_reason"] == "STOP_LOSS").sum() if "exit_reason" in trades.columns else 0
    stop_rate    = stop_hits / n_trades * 100 if n_trades > 0 else 0

    return {
        "Total Return (%)":  total_return,
        "Final Equity ($)":  final,
        "Max Drawdown (%)":  max_dd,
        "Sharpe Ratio":      sharpe,
        "Total Trades":      n_trades,
        "Win Rate (%)":      win_rate,
        "Profit Factor":     pf,
        "Stop Hit Rate (%)": stop_rate,
        "drawdown":          drawdown,
    }


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
def plot_summary(all_results, all_stats):
    GOLD  = "#c9a84c"; GREEN = "#3ddc97"; RED  = "#ff5f6d"
    BLUE  = "#4fa3e0"; DIM   = "#888899"; BG   = "#0d0d0f"; PANEL = "#14141a"

    n = len(all_results)
    if n == 0:
        print("Nothing to plot.")
        return

    window_str = all_results[0].get("window", "unknown")
    fig = plt.figure(figsize=(18, 5 + n * 2.2), facecolor=BG)
    fig.text(0.5, 0.98, "NSAI ENGINE  ·  BEST CONFIGURATION",
             ha="center", fontsize=14, color=GOLD,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.965,
             f"Z={Z_THRESHOLD} | Mode={SIZING_MODE} | ${DOLLAR_SIZE:,.0f}/leg | "
             f"Stop={STOP_LOSS_PCT*100:.0f}% | Gate={'ON' if ROLLING_COINT_GATE else 'OFF'} | "
             f"Window: {window_str}",
             ha="center", fontsize=9, color=DIM, fontfamily="monospace")

    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.6, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)
    dollar_fmt = FuncFormatter(lambda x, _: f"${x:,.0f}")

    for row, (result, stats) in enumerate(zip(all_results, all_stats)):
        label = f"{result['t1']}/{result['t2']}"
        eq    = result["equity"]
        trades= result["trades"]
        base  = result["starting_cash"]
        color = GREEN if result["final_cash"] >= base else RED
        beta  = result.get("ols_beta", 0)

        ax_eq = fig.add_subplot(gs[row, 0])
        ax_eq.set_facecolor(PANEL)
        ax_eq.tick_params(colors=DIM, labelsize=7)
        for sp in ax_eq.spines.values(): sp.set_edgecolor("#2a2a35")
        ax_eq.plot(eq, color=color, lw=1.0)
        ax_eq.fill_between(range(len(eq)), eq, base,
                            where=np.array(eq) >= base, color=GREEN, alpha=0.1)
        ax_eq.fill_between(range(len(eq)), eq, base,
                            where=np.array(eq) < base, color=RED, alpha=0.12)
        ax_eq.axhline(base, color=DIM, lw=0.5, ls="--")
        ax_eq.yaxis.set_major_formatter(dollar_fmt)
        ax_eq.set_title(f"{label}  [OLS β={beta:.3f}]",
                        fontsize=8, color=GOLD, fontfamily="monospace", loc="left")

        ax_pnl = fig.add_subplot(gs[row, 1])
        ax_pnl.set_facecolor(PANEL)
        ax_pnl.tick_params(colors=DIM, labelsize=7)
        for sp in ax_pnl.spines.values(): sp.set_edgecolor("#2a2a35")
        if len(trades):
            cols = [GREEN if p > 0 else RED for p in trades["pnl"]]
            ax_pnl.bar(range(len(trades)), trades["pnl"], color=cols, width=0.7)
            ax_pnl.axhline(0, color=DIM, lw=0.5)
            ax_pnl.yaxis.set_major_formatter(dollar_fmt)
        ax_pnl.set_title("Trade PnL", fontsize=8, color=DIM,
                          fontfamily="monospace", loc="left")

        ax_st = fig.add_subplot(gs[row, 2])
        ax_st.set_facecolor(PANEL)
        for sp in ax_st.spines.values(): sp.set_edgecolor("#2a2a35")
        ax_st.axis("off")
        ret_c = GREEN if stats["Total Return (%)"] > 0 else RED
        wr_c  = GREEN if stats["Win Rate (%)"] > 50   else RED
        pf_c  = GREEN if stats["Profit Factor"] > 1   else RED
        rows = [
            ("Return:  ", f"{stats['Total Return (%)']:+.2f}%",      ret_c),
            ("Trades:  ", f"{stats['Total Trades']}",                 GOLD),
            ("Win Rate:", f"{stats['Win Rate (%)']:.0f}%",            wr_c),
            ("Sharpe:  ", f"{stats['Sharpe Ratio']:.2f}",             BLUE),
            ("PF:      ", f"{stats['Profit Factor']:.2f}",            pf_c),
            ("MaxDD:   ", f"{stats['Max Drawdown (%)']:.2f}%",        RED),
            ("StopHit: ", f"{stats['Stop Hit Rate (%)']:.0f}%",       DIM),
        ]
        for mi, (lbl, val, col) in enumerate(rows):
            y = 0.95 - mi * 0.135
            ax_st.text(0.02, y, lbl, transform=ax_st.transAxes,
                       fontsize=8.5, color=DIM, fontfamily="monospace", va="top")
            ax_st.text(0.99, y, val, transform=ax_st.transAxes,
                       fontsize=8.5, color=col, fontfamily="monospace",
                       va="top", ha="right", fontweight="bold")

    save = SAVE_PATH.replace(".png", "_best_config.png")
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"[saved] {save}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run_all():
    all_results = []
    all_stats   = []

    for t1, t2 in PAIRS:
        print(f"\n[+] {t1}/{t2}")
        result = run_backtest(t1, t2)
        if result is None:
            continue
        stats = compute_stats(result)
        all_results.append(result)
        all_stats.append(stats)
        print(f"    {stats['Total Trades']} trades | "
              f"Return: {stats['Total Return (%)']:+.2f}% | "
              f"Sharpe: {stats['Sharpe Ratio']:.2f} | "
              f"WinRate: {stats['Win Rate (%)']:.0f}% | "
              f"PF: {stats['Profit Factor']:.2f}")

    if not all_results:
        print("No results generated.")
        return

    plot_summary(all_results, all_stats)


if __name__ == "__main__":
    run_all()
