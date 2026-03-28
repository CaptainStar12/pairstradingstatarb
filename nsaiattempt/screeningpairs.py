import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── Candidate universe ────────────────────────────────────────────────────────
# Grouped by sector. The scanner tests every pair within each sector group.
# Pairs from different sectors are unlikely to be cointegrated.
# Add or remove tickers freely — more tickers = more candidate pairs tested.

UNIVERSE = {
    "Banks": [
        "JPM", "BAC", "WFC", "USB", "PNC", "TFC", "CFG", "FITB", "HBAN", "RF"
    ],
    "Investment Banks": [
        "GS", "MS", "BLK", "SCHW", "AMP"
    ],
    "Payments": [
        "V", "MA", "AXP", "COF", "SYF"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "MU", "QCOM", "AVGO", "TXN"
    ],
    "Oil Majors": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO"
    ],
    "Consumer Staples": [
        "KO", "PEP", "PM", "MO", "STZ", "KHC", "GIS", "CPB"
    ],
    "Big Tech": [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN"
    ],
    "Telecom": [
        "T", "VZ", "TMUS"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP", "EXC", "XEL", "ES"
    ],
    "Healthcare": [
        "JNJ", "ABT", "MDT", "BSX", "SYK", "ZBH"
    ],
    "Pharma": [
        "PFE", "MRK", "BMY", "ABBV", "LLY", "AMGN"
    ],
    "Airlines": [
        "DAL", "UAL", "AAL", "LUV", "ALK"
    ],
    "Retail": [
        "WMT", "TGT", "COST", "KR", "DG", "DLTR"
    ],
}

# ── Screening parameters ──────────────────────────────────────────────────────
COINT_YEARS        = 3      # years of daily data for structural test
BETA_MIN           = 0.3   # OLS beta below this → pair skipped (too small)
BETA_MAX           = 3.0   # OLS beta above this → pair skipped (too extreme)
DAILY_PVALUE       = 0.10   # Stage 1 threshold — strict on daily
INTRADAY_PVALUE    = 0.25   # Stage 2 threshold — looser on intraday
INTRADAY_BARS      = 200    # bars used for intraday test
MIN_DAILY_BARS     = 200    # minimum daily bars required
TOP_N              = 20     # show top N pairs in final ranking

# ── Data cache — download each ticker once ────────────────────────────────────
_daily_cache   = {}
_intraday_cache = {}

def get_daily(ticker):
    if ticker not in _daily_cache:
        try:
            d = yf.download(ticker, period=f"{COINT_YEARS}y", interval="1d",
                            auto_adjust=True, progress=False)
            close = d["Close"]
            # yfinance sometimes returns DataFrame with MultiIndex columns
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            _daily_cache[ticker] = close.dropna()
        except Exception:
            _daily_cache[ticker] = pd.Series(dtype=float)
    return _daily_cache[ticker]

def get_intraday(ticker):
    if ticker not in _intraday_cache:
        try:
            d = yf.download(ticker, period="60d", interval="5m",
                            auto_adjust=True, progress=False)
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            _intraday_cache[ticker] = close.dropna()
        except Exception:
            _intraday_cache[ticker] = pd.Series(dtype=float)
    return _intraday_cache[ticker]


# ── Per-pair cointegration test ───────────────────────────────────────────────
def test_pair(t1, t2):
    """
    Run dual cointegration screen on one pair.
    Returns dict with all results, or None if data insufficient.
    """
    s1_daily = get_daily(t1)
    s2_daily = get_daily(t2)

    # Align daily series
    daily = pd.merge(s1_daily.rename("P1"), s2_daily.rename("P2"),
                     left_index=True, right_index=True).dropna()
    if len(daily) < MIN_DAILY_BARS:
        return None

    # Stage 1 — daily cointegration + OLS beta
    beta, alpha = 1.0, 0.0
    try:
        _, daily_pval, _ = coint(daily["P1"].values, daily["P2"].values)
        beta, alpha = np.polyfit(daily["P2"].values, daily["P1"].values, 1)
    except Exception:
        return None

    # Beta sanity check — skip pairs with extreme hedge ratios
    if not (BETA_MIN <= abs(beta) <= BETA_MAX):
        return None

    # Stage 2 — intraday cointegration
    s1_intra = get_intraday(t1)
    s2_intra = get_intraday(t2)
    intra = pd.merge(s1_intra.rename("P1"), s2_intra.rename("P2"),
                     left_index=True, right_index=True).dropna()

    if len(intra) < INTRADAY_BARS:
        intraday_pval = 1.0
    else:
        try:
            _, intraday_pval, _ = coint(
                intra["P1"].values[:INTRADAY_BARS],
                intra["P2"].values[:INTRADAY_BARS]
            )
        except Exception:
            intraday_pval = 1.0

    # ADF on intraday spread as supplementary check
    try:
        spread = intra["P1"].values - (beta * intra["P2"].values + alpha)
        adf_pval = adfuller(spread[:INTRADAY_BARS], autolag="AIC")[1] if len(spread) >= INTRADAY_BARS else 1.0
    except Exception:
        adf_pval = 1.0

    daily_pass    = daily_pval    < DAILY_PVALUE
    intraday_pass = intraday_pval < INTRADAY_PVALUE

    # Composite score — lower is better
    # Rewards pairs that pass both screens with low p-values
    score = daily_pval * 0.6 + intraday_pval * 0.4

    return {
        "pair":          f"{t1}/{t2}",
        "t1":            t1,
        "t2":            t2,
        "daily_p":       daily_pval,
        "intraday_p":    intraday_pval,
        "adf_p":         adf_pval,
        "beta":          beta,
        "alpha":         alpha,
        "daily_pass":    daily_pass,
        "intraday_pass": intraday_pass,
        "both_pass":     daily_pass and intraday_pass,
        "score":         score,
        "daily_bars":    len(daily),
        "intraday_bars": len(intra),
    }


# ── Main scanner ──────────────────────────────────────────────────────────────
def run_scanner():
    all_tickers = sorted(set(t for group in UNIVERSE.values() for t in group))

    print(f"\n{'='*65}")
    print(f"  NSAI PAIR SCANNER")
    print(f"  Universe: {len(all_tickers)} tickers across {len(UNIVERSE)} sectors")
    print(f"  Daily threshold: p<{DAILY_PVALUE}  |  Intraday threshold: p<{INTRADAY_PVALUE}")
    print(f"{'='*65}")

    # Pre-download all tickers
    print(f"\n[1/3] Downloading daily data ({COINT_YEARS}yr) for {len(all_tickers)} tickers...")
    for i, t in enumerate(all_tickers):
        get_daily(t)
        print(f"  {i+1:>3}/{len(all_tickers)}  {t}", end="\r")
    print(f"  Daily data cached.                    ")

    print(f"\n[2/3] Downloading intraday data (60d 5m) for {len(all_tickers)} tickers...")
    for i, t in enumerate(all_tickers):
        get_intraday(t)
        print(f"  {i+1:>3}/{len(all_tickers)}  {t}", end="\r")
    print(f"  Intraday data cached.                 ")

    # Generate all within-sector pairs
    candidate_pairs = []
    for sector, tickers in UNIVERSE.items():
        for t1, t2 in combinations(tickers, 2):
            candidate_pairs.append((t1, t2, sector))

    print(f"\n[3/3] Testing {len(candidate_pairs)} candidate pairs...")
    results = []
    passed  = []

    for i, (t1, t2, sector) in enumerate(candidate_pairs):
        print(f"  {i+1:>3}/{len(candidate_pairs)}  {t1}/{t2}        ", end="\r")
        r = test_pair(t1, t2)
        if r is None:
            continue
        r["sector"] = sector
        results.append(r)
        if r["both_pass"]:
            passed.append(r)

    print(f"  Done. {len(results)} pairs tested.              ")

    # ── Results ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values("score")

    print(f"\n{'='*65}")
    print(f"  PAIRS PASSING BOTH SCREENS  ({len(passed)} found)")
    print(f"{'='*65}")
    if passed:
        passed_df = pd.DataFrame(passed).sort_values("score")
        print(f"  {'Pair':<12} {'Sector':<22} {'Daily p':>8} {'Intra p':>8} {'ADF p':>8} {'Beta':>6}")
        print(f"  {'-'*68}")
        for _, r in passed_df.iterrows():
            print(f"  {r['pair']:<12} {r['sector']:<22} "
                  f"{r['daily_p']:>8.4f} {r['intraday_p']:>8.4f} {r['adf_p']:>8.4f} {r['beta']:>6.3f}")
    else:
        print("  None found at current thresholds.")
        print(f"  Try: increase DAILY_PVALUE to 0.10 or INTRADAY_PVALUE to 0.20")

    print(f"\n{'='*65}")
    print(f"  TOP {TOP_N} PAIRS BY COMPOSITE SCORE  (regardless of pass/fail)")
    print(f"{'='*65}")
    print(f"  {'Pair':<12} {'Sector':<22} {'Daily p':>8} {'Intra p':>8} {'Pass?':>6}")
    print(f"  {'-'*60}")
    for _, r in df.head(TOP_N).iterrows():
        status = "BOTH" if r["both_pass"] else ("D only" if r["daily_pass"] else ("I only" if r["intraday_pass"] else "neither"))
        print(f"  {r['pair']:<12} {r['sector']:<22} "
              f"{r['daily_p']:>8.4f} {r['intraday_p']:>8.4f} {status:>6}")

    print(f"\n{'='*65}")
    print(f"  BEST PAIRS PER SECTOR")
    print(f"{'='*65}")
    for sector in UNIVERSE:
        sector_df = df[df["sector"] == sector].head(3)
        if len(sector_df) == 0:
            continue
        best = sector_df.iloc[0]
        marker = " ✓" if best["both_pass"] else ""
        print(f"  {sector:<22}  {best['pair']:<12}  "
              f"daily={best['daily_p']:.4f}  intra={best['intraday_p']:.4f}{marker}")

    print(f"\n{'='*65}")
    if passed:
        print(f"  TO TRADE: copy passing pairs into backtest_nsai.py PAIRS list")
        print(f"  Example:")
        for r in passed[:3]:
            print(f"    (\"{r['t1']}\", \"{r['t2']}\"),")
    print(f"{'='*65}\n")

    return df, passed


if __name__ == "__main__":
    results_df, passing_pairs = run_scanner()
