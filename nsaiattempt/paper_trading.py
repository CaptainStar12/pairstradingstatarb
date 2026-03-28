"""
NSAI Paper Trading Engine — Interactive Brokers (IB Gateway)
=============================================================
Runs the NSAI pairs trading strategy in real time using IBKR's
paper trading account via IB Gateway and ib_insync.

Best configuration (matched to backtest that produced +1.82%, Sharpe 1.99):
  Z threshold:   1.5
  Dollar size:   $50,000/leg
  Stop loss:     1% of combined notional ($1,000 max per trade)
  Exit threshold: Z < 0.1 (full reversion)
  Timeout:       50 bars (~4 hours)
  Rolling coint gate: 500 bars, p < 0.10

Setup:
1. pip install ib_insync
2. Open IB Gateway, log in to PAPER trading account
3. Configure → Settings → API:
      Socket port: 7497
      Read-Only API: UNCHECKED
      Allow connections from localhost: CHECKED
4. python paper_trading.py

Run this script before 9:30am ET each trading day.
Everything is logged to nsai_paper_trading.log.
"""

from ib_insync import IB, Stock, MarketOrder, util
from statsmodels.tsa.stattools import coint
from nsaibrain import NSAIEngine
import numpy as np
import pandas as pd
import datetime
import logging
import time

# ── IBKR connection ───────────────────────────────────────────────────────────
IB_HOST   = "127.0.0.1"
IB_PORT   = 7497           # 7497 = paper, 7496 = live — DO NOT use 7496 yet
IB_CLIENT = 1

# ── Strategy (matched exactly to best backtest configuration) ─────────────────
T1 = "JPM"
T2 = "BAC"

Z_THRESHOLD          = 1.5
DOLLAR_SIZE          = 50000.0
STOP_LOSS_PCT        = 0.01
ROLLING_COINT_BARS   = 500
ROLLING_COINT_PVALUE = 0.10
WARMUP_BARS          = 100
BAR_SIZE             = "5 mins"
EXCHANGE             = "SMART"
CURRENCY             = "USD"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("nsai_paper_trading.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("NSAI")


# ─────────────────────────────────────────────
#  CONNECTION
# ─────────────────────────────────────────────
def connect_ibkr():
    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT)
        acc = ib.accountValues()
        net = next((x.value for x in acc
                    if x.tag == "NetLiquidation" and x.currency == "USD"), "N/A")
        log.info(f"Connected to IB Gateway — Account NLV: ${float(net):,.2f}")
        return ib
    except Exception as e:
        log.error(f"Connection failed: {e}")
        log.error(f"Make sure IB Gateway is running on port {IB_PORT} "
                  f"with paper trading account.")
        raise


def make_contract(symbol):
    return Stock(symbol, EXCHANGE, CURRENCY)


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
def get_bars(ib, symbol, n_bars=600):
    contract = make_contract(symbol)
    ib.qualifyContracts(contract)
    days = max(2, int(n_bars / 78) + 2)
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=f"{days} D",
            barSizeSetting=BAR_SIZE,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return pd.Series(dtype=float)
        df = util.df(bars)
        s  = df["close"].dropna()
        return s.iloc[-n_bars:] if len(s) > n_bars else s
    except Exception as e:
        log.error(f"get_bars({symbol}): {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────
#  SIZING
# ─────────────────────────────────────────────
def compute_shares(p1, p2, beta):
    beta_clipped = np.clip(abs(beta), 0.1, 10.0)
    s1 = max(1, int(DOLLAR_SIZE / p1))
    s2 = max(1, int(beta_clipped * DOLLAR_SIZE / p2))
    return s1, s2


# ─────────────────────────────────────────────
#  ORDERS
# ─────────────────────────────────────────────
def submit_order(ib, symbol, qty, side):
    contract = make_contract(symbol)
    ib.qualifyContracts(contract)
    trade = ib.placeOrder(contract, MarketOrder(side, qty))
    ib.sleep(1)
    log.info(f"  ORDER {side} {qty} {symbol} — {trade.orderStatus.status}")
    return trade


def close_all(ib):
    for pos in ib.positions():
        symbol = pos.contract.symbol
        qty    = abs(int(pos.position))
        side   = "SELL" if pos.position > 0 else "BUY"
        log.info(f"EOD close: {side} {qty} {symbol}")
        submit_order(ib, symbol, qty, side)


# ─────────────────────────────────────────────
#  MARKET HOURS
# ─────────────────────────────────────────────
def is_market_open():
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=15, minute=55, second=0, microsecond=0)
    return open_ <= now <= close_


def wait_for_open():
    if is_market_open():
        return
    log.info("Waiting for market open (09:30 ET)...")
    while not is_market_open():
        time.sleep(60)
    log.info("Market open.")


def secs_to_next_bar():
    now = datetime.datetime.now()
    return max(300 - (now.minute % 5 * 60 + now.second) + 5, 10)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    log.info("="*60)
    log.info("  NSAI PAPER TRADING — IBKR IB Gateway")
    log.info(f"  {T1}/{T2} | Z={Z_THRESHOLD} | ${DOLLAR_SIZE:,.0f}/leg | "
             f"Stop={STOP_LOSS_PCT*100:.0f}% | Gate={ROLLING_COINT_BARS}bars")
    log.info("="*60)

    ib = connect_ibkr()
    wait_for_open()

    # Fetch history for warmup and initial rolling buffer
    log.info("Fetching historical data for warmup...")
    n_fetch = WARMUP_BARS + ROLLING_COINT_BARS + 20
    s1_hist = get_bars(ib, T1, n_bars=n_fetch)
    s2_hist = get_bars(ib, T2, n_bars=n_fetch)

    if len(s1_hist) < WARMUP_BARS or len(s2_hist) < WARMUP_BARS:
        log.error("Insufficient historical data. Is IB Gateway connected?")
        ib.disconnect()
        return

    hist = pd.merge(s1_hist.rename("P1"), s2_hist.rename("P2"),
                    left_index=True, right_index=True).dropna()
    p1v, p2v = hist["P1"].values, hist["P2"].values

    # OLS initialization
    r1 = np.diff(np.log(p1v))
    r2 = np.diff(np.log(p2v))
    rb, _       = np.polyfit(r2, r1, 1)
    ols_beta    = rb * (p1v.mean() / p2v.mean())
    ols_alpha   = p1v.mean() - ols_beta * p2v.mean()
    log.info(f"OLS beta={ols_beta:.4f}, alpha={ols_alpha:.4f}")

    # Create and warm up engine
    engine = NSAIEngine(require_warmup=True,
                        initial_beta=ols_beta,
                        initial_alpha=ols_alpha)
    engine.warmup(p1v[-WARMUP_BARS:], p2v[-WARMUP_BARS:])
    log.info("Engine warmed up. Entering trading loop.")

    # State
    pos           = 0
    entry_p1      = entry_p2 = 0.0
    entry_s1      = entry_s2 = 0
    p1_buf        = list(p1v[-ROLLING_COINT_BARS:])
    p2_buf        = list(p2v[-ROLLING_COINT_BARS:])
    bar_count     = 0
    stop_amt      = DOLLAR_SIZE * 2 * STOP_LOSS_PCT

    # Evaluate gate immediately using pre-filled historical buffer
    # The buffer is already full from historical data so there is no
    # waiting period — the gate reflects the current cointegration state
    # from the very first live bar.
    try:
        _, _pv, _ = coint(np.array(p1_buf), np.array(p2_buf))
        gate_open = _pv < ROLLING_COINT_PVALUE
        log.info(f"Initial gate evaluation: p={_pv:.4f} → "
                 f"{'OPEN' if gate_open else 'CLOSED'}")
    except Exception:
        gate_open = False
        log.info("Initial gate evaluation failed — gate CLOSED")

    try:
        while is_market_open():
            ib.sleep(0)

            s1 = get_bars(ib, T1, n_bars=2)
            s2 = get_bars(ib, T2, n_bars=2)
            if s1.empty or s2.empty:
                time.sleep(30)
                continue

            p1 = float(s1.iloc[-1])
            p2 = float(s2.iloc[-1])
            bar_count += 1

            # Update buffers
            p1_buf.append(p1); p2_buf.append(p2)
            if len(p1_buf) > ROLLING_COINT_BARS:
                p1_buf.pop(0); p2_buf.pop(0)

            # Rolling cointegration gate
            if len(p1_buf) >= ROLLING_COINT_BARS:
                try:
                    _, pv, _ = coint(np.array(p1_buf), np.array(p2_buf))
                    gate_open = pv < ROLLING_COINT_PVALUE
                except Exception:
                    gate_open = False

            # Signal
            res    = engine.get_signal(p1, p2, current_pos=pos,
                                       z_threshold=Z_THRESHOLD)
            action = res["action"]
            z      = res["z"]
            beta   = res["beta"]

            log.info(f"bar={bar_count:>4}  {T1}=${p1:.2f}  {T2}=${p2:.2f}  "
                     f"z={z:+.3f}  gate={'OPEN' if gate_open else 'CLOSED'}  "
                     f"pos={pos:+d}  {action}")

            # Stop loss
            if pos != 0:
                unrl = pos*entry_s1*(p1-entry_p1) - pos*entry_s2*(p2-entry_p2)
                if unrl < -stop_amt:
                    log.info(f"STOP LOSS: ${unrl:.2f}")
                    action = "STOP_LOSS"

            # Entry
            if action in ("LONG", "SHORT") and pos == 0 and gate_open:
                ns1, ns2 = compute_shares(p1, p2, beta)
                if action == "LONG":
                    submit_order(ib, T1, ns1, "BUY")
                    submit_order(ib, T2, ns2, "SELL")
                    pos = 1
                else:
                    submit_order(ib, T1, ns1, "SELL")
                    submit_order(ib, T2, ns2, "BUY")
                    pos = -1
                entry_p1, entry_p2 = p1, p2
                entry_s1, entry_s2 = ns1, ns2
                log.info(f"ENTERED {action}: {ns1}×{T1} / {ns2}×{T2}")

            # Exit
            elif action in ("EXIT", "EXIT_TIMEOUT", "STOP_LOSS") and pos != 0:
                pnl = pos*entry_s1*(p1-entry_p1) - pos*entry_s2*(p2-entry_p2)
                if pos == 1:
                    submit_order(ib, T1, entry_s1, "SELL")
                    submit_order(ib, T2, entry_s2, "BUY")
                else:
                    submit_order(ib, T1, entry_s1, "BUY")
                    submit_order(ib, T2, entry_s2, "SELL")
                log.info(f"EXITED ({action}): PnL=${pnl:.2f}")
                pos = 0

            time.sleep(secs_to_next_bar())

    except KeyboardInterrupt:
        log.info("Stopped by user.")
    finally:
        if pos != 0:
            log.info("Closing open positions...")
            close_all(ib)
        ib.disconnect()
        log.info("Disconnected. Session ended.")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  NSAI Paper Trading — IBKR IB Gateway")
    print("="*60)
    print(f"  Connecting to IB Gateway at {IB_HOST}:{IB_PORT}")
    print(f"  Ensure IB Gateway is running on PAPER account.")
    print(f"  Press Ctrl+C to stop safely at any time.")
    print("="*60 + "\n")
    run()
