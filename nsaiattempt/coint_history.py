import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
T1 = "JPM"
T2 = "BAC"

# Daily data window for rolling cointegration test
COINT_YEARS   = 5        # how many years of daily data to download
ROLL_WINDOW   = 63       # rolling window in trading days (~3 months)
PVALUE_THRESH = 0.10     # cointegration significance threshold

# Hourly data for intraday cointegration check at each window
# Set to True to also test intraday cointegration at each date
# (slower but more informative)
CHECK_INTRADAY = False

# ─────────────────────────────────────────────
#  DOWNLOAD
# ─────────────────────────────────────────────
print(f"\nDownloading {COINT_YEARS} years of daily data for {T1} and {T2}...")
d1 = yf.download(T1, period=f"{COINT_YEARS}y", interval="1d",
                 auto_adjust=True, progress=False)
d2 = yf.download(T2, period=f"{COINT_YEARS}y", interval="1d",
                 auto_adjust=True, progress=False)

# Fix yfinance DataFrame/Series issue
def fix_close(raw):
    c = raw["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c.dropna()

s1 = fix_close(d1)
s2 = fix_close(d2)

df = pd.merge(s1.rename("P1"), s2.rename("P2"),
              left_index=True, right_index=True).dropna()

print(f"Got {len(df)} daily bars from {df.index[0].date()} to {df.index[-1].date()}")

# ─────────────────────────────────────────────
#  ROLLING COINTEGRATION
# ─────────────────────────────────────────────
print(f"\nRunning rolling cointegration (window={ROLL_WINDOW} days)...")

dates      = []
pvalues    = []
betas      = []
adf_pvals  = []
is_coint   = []

for end in range(ROLL_WINDOW, len(df)):
    window = df.iloc[end - ROLL_WINDOW:end]
    p1w    = window["P1"].values
    p2w    = window["P2"].values

    try:
        _, pval, _ = coint(p1w, p2w)
        beta, alpha = np.polyfit(p2w, p1w, 1)

        # ADF on the residual spread
        spread  = p1w - (beta * p2w + alpha)
        adf_p   = adfuller(spread, autolag="AIC")[1]

    except Exception:
        pval  = 1.0
        beta  = np.nan
        adf_p = 1.0

    dates.append(df.index[end])
    pvalues.append(pval)
    betas.append(beta)
    adf_pvals.append(adf_p)
    is_coint.append(pval < PVALUE_THRESH)

results = pd.DataFrame({
    "date":      dates,
    "pvalue":    pvalues,
    "beta":      betas,
    "adf_pval":  adf_pvals,
    "coint":     is_coint,
}).set_index("date")

# ─────────────────────────────────────────────
#  FIND COINTEGRATED PERIODS
# ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ROLLING COINTEGRATION RESULTS  [{T1}/{T2}]")
print(f"  Window: {ROLL_WINDOW} days | Threshold: p<{PVALUE_THRESH}")
print(f"{'='*65}")

# Find contiguous cointegrated periods
in_period   = False
period_start = None
periods     = []

for date, row in results.iterrows():
    if row["coint"] and not in_period:
        in_period    = True
        period_start = date
    elif not row["coint"] and in_period:
        in_period = False
        periods.append((period_start, date,
                        (date - period_start).days))

if in_period:
    periods.append((period_start, results.index[-1],
                    (results.index[-1] - period_start).days))

pct_coint = results["coint"].mean() * 100
print(f"\n  Cointegrated {pct_coint:.1f}% of the time\n")

if periods:
    print(f"  {'Start':<14} {'End':<14} {'Days':>6} {'Avg p':>8} {'Avg β':>8}")
    print(f"  {'-'*52}")
    for start, end, days in sorted(periods, key=lambda x: -x[2]):
        window_data = results.loc[start:end]
        avg_p    = window_data["pvalue"].mean()
        avg_beta = window_data["beta"].mean()
        marker   = " ← BEST" if days == max(p[2] for p in periods) else ""
        print(f"  {str(start.date()):<14} {str(end.date()):<14} "
              f"{days:>6} {avg_p:>8.4f} {avg_beta:>8.3f}{marker}")
else:
    print("  No cointegrated periods found at this threshold.")
    print(f"  Try increasing PVALUE_THRESH (currently {PVALUE_THRESH})")

print(f"\n  Current p-value (most recent window): "
      f"{results['pvalue'].iloc[-1]:.4f}")
print(f"  Current beta:                         "
      f"{results['beta'].iloc[-1]:.4f}")

# ─────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────
BG    = "#0d0d0f"; PANEL = "#14141a"; GOLD  = "#c9a84c"
GREEN = "#3ddc97"; RED   = "#ff5f6d"; DIM   = "#888899"; BLUE = "#4fa3e0"

fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=BG)
fig.suptitle(f"{T1}/{T2}  ·  Rolling {ROLL_WINDOW}-Day Cointegration",
             color=GOLD, fontsize=13, fontfamily="monospace", fontweight="bold")

# Panel 1 — price series
ax1 = axes[0]
ax1.set_facecolor(PANEL)
ax1.tick_params(colors=DIM, labelsize=7)
for sp in ax1.spines.values(): sp.set_edgecolor("#2a2a35")
ax2_twin = ax1.twinx()
ax2_twin.set_facecolor(PANEL)
ax2_twin.tick_params(colors=DIM, labelsize=7)
ax1.plot(df.index, df["P1"], color=GREEN, lw=0.8, label=T1)
ax2_twin.plot(df.index, df["P2"], color=BLUE, lw=0.8, label=T2)
ax1.set_ylabel(T1, color=GREEN, fontsize=8)
ax2_twin.set_ylabel(T2, color=BLUE, fontsize=8)
ax1.set_title("Price Series", color=DIM, fontsize=8,
              fontfamily="monospace", loc="left")

# Panel 2 — rolling p-value with threshold line and shaded cointegrated regions
ax2 = axes[1]
ax2.set_facecolor(PANEL)
ax2.tick_params(colors=DIM, labelsize=7)
for sp in ax2.spines.values(): sp.set_edgecolor("#2a2a35")
ax2.plot(results.index, results["pvalue"], color=GOLD, lw=0.8)
ax2.axhline(PVALUE_THRESH, color=RED, lw=0.8, ls="--",
            label=f"p={PVALUE_THRESH} threshold")
# Shade cointegrated periods
for start, end, _ in periods:
    ax2.axvspan(start, end, alpha=0.15, color=GREEN)
ax2.set_ylabel("p-value", color=DIM, fontsize=8)
ax2.set_ylim(0, 1)
ax2.set_title("Rolling Cointegration p-value  (green = cointegrated)",
              color=DIM, fontsize=8, fontfamily="monospace", loc="left")
ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=DIM)

# Panel 3 — rolling beta
ax3 = axes[2]
ax3.set_facecolor(PANEL)
ax3.tick_params(colors=DIM, labelsize=7)
for sp in ax3.spines.values(): sp.set_edgecolor("#2a2a35")
ax3.plot(results.index, results["beta"], color=BLUE, lw=0.8)
ax3.axhline(0, color=DIM, lw=0.5, ls="--")
# Shade cointegrated periods
for start, end, _ in periods:
    ax3.axvspan(start, end, alpha=0.15, color=GREEN)
ax3.set_ylabel("OLS Beta", color=DIM, fontsize=8)
ax3.set_title("Rolling OLS Beta  (hedge ratio over time)",
              color=DIM, fontsize=8, fontfamily="monospace", loc="left")

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("jpm_bac_cointegration_history.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("\n[saved] jpm_bac_cointegration_history.png")
print("="*65)
print("\nTo backtest a specific cointegrated period, use:")
print('  START_DATE, END_DATE, INTERVAL, WARMUP_BARS = ')
print('  "YYYY-MM-DD", "YYYY-MM-DD", "1h", 30')
print("  in backtest_nsai.py\n")
