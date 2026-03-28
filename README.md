NSAI Pairs Trading Engine
A statistical pairs trading engine combing a Kalman filter for dynamic hedge ratio estimation, a Hurst exponent regime filter, and a rolling cointegration gate. Built as a research project to explore mean-reversion signal extraction from intraday equity price data.

What It Does
The engine trades the spread between two cointegrated stocks. When the spread deviates significantly from its expected value, the engine enters a position betting on reversion. It exits when the spread returns to almost zero or after a max holding period of time. 

The core signal is the Kalman innovation Z-score. That Kalman innovation Z-score is the difference between what the Kalman filter predicted the spread would be and what it really was, divided by the expected standard deviation of that prediction. A large Z-score means the spread is more extended than the filter expected. This means that the Z-score is a statistically grounded signal for reversion. 

Architecture
Signal Pipeline
Raw Prices
    → OLS Regression (returns-based beta/alpha initialization)
    → Kalman Filter (dynamic hedge ratio tracking)
    → Innovation Z-Score (entry/exit signal)
    → Hurst Exponent Gate (regime filter: H < 0.5 = mean-reverting)
    → Rolling Cointegration Gate (pair validity: p < 0.10)
    → Dollar-Neutral Position Sizing
    → 1-Bar Execution Delay (realistic fill simulation)
Key Components
nsaibrain.py — NSAIEngine class
The core engine. Runs a 2-state Kalman filter tracking beta (hedge ratio) and alpha (intercept) between two price series. Computes the standardized innovation Z-score as the trading signal. The Hurst exponent filter prevents entries during trending regimes.
backtest_nsai.py — Backtesting framework
Downloads data via yfinance, runs dual cointegration screening, warms up the engine, and simulates trading with realistic execution delay and fees. Produces a dashboard chart with equity curve, trade PnL, and performance statistics.
screeningpairs.py — Pair scanner
Tests all within-sector pairs across a configurable universe of 80+ tickers. Runs dual cointegration screening (daily structural test + intraday timescale test) and ranks pairs by composite score. Filters extreme hedge ratios.
coint_history.py — Rolling cointegration analyzer
Downloads 5 years of daily data for a pair and computes rolling 63-day cointegration p-values. Identifies all cointegrated periods historically and produces a chart showing when the relationship was strong or broken.
paper_trading.py — Live execution via IBKR
Connects to Interactive Brokers via IB Gateway using ib_insync. Fetches historical bars for warmup, evaluates the rolling cointegration gate from startup, and executes paper trades automatically during market hours.

Kalman Filter Design
The filter tracks a 2-dimensional state vector x = [beta, alpha] where beta is the dynamic hedge ratio and alpha is the intercept. The measurement model is:
p1 = beta * p2 + alpha + noise
At each bar the filter:

Predicts — adds process noise Q to uncertainty P, expecting slow drift in the relationship
Updates — computes the innovation y = p1 - H @ x, calculates Kalman gain K, and updates state and covariance

The innovation y divided by sqrt(S) (innovation variance) is the Z-score signal. This is theoretically standard normal when the filter model is correct, giving the signal a principled statistical interpretation.
OLS Initialization: Rather than starting at [0, 0], beta and alpha are seeded from a returns-based OLS regression on 3 years of daily data. This prevents the cold-start problem where the filter spends warmup bars recovering from a catastrophically wrong prior.

Regime Filtering
Two independent filters gate entries:
Hurst Exponent (bar-by-bar): Computed via rescaled range analysis on the last 50 bars of spread history. Values below 0.5 indicate mean-reversion. Only enters when H < 0.5. This prevents trading during trending regimes where the spread is unlikely to revert.
Rolling Cointegration Gate (bar-by-bar): Runs the Engle-Granger cointegration test on the last 500 bars of intraday prices every bar. Only allows new entries when p < 0.10. This prevents trading during periods when the structural relationship between the pair has broken down. The buffer is pre-filled from historical data at startup — no waiting period required.

Position Sizing
Dollar-neutral sizing ensures both legs have equal notional exposure at entry:
shares_p1 = DOLLAR_SIZE / price_p1
shares_p2 = beta * DOLLAR_SIZE / price_p2
The Kalman beta is clipped to [0.1, 10.0] to prevent extreme sizing from an unstable early estimate. At $50,000/leg the $10 round-trip fee is approximately 0.01% of position — negligible relative to the signal edge.

Stop Loss
A hard dollar stop exits any position where unrealized PnL falls below -1% of combined notional ($1,000 at $50k/leg). This is appropriate for mean-reversion. This stop aims for catastrophically non-reverting trades, not normal reversion-in-progress moves. Tight stops are counterproductive in mean-reversion because a move against you means the spread is more extended, not that the thesis is wrong.

Research Findings
Signal validity: Regression of the Kalman innovation Z-score against next-bar spread changes on JPM/BAC produced p < 0.0001 with monotone quintile analysis and 95% directional accuracy. The signal is statistically real.
Pair selection: JPM/BAC was cointegrated approximately 14% of the time over the past 5 years based on rolling 63-day windows. The pair works at 5-minute resolution during cointegrated periods and does not work at hourly resolution.
Dollar neutrality: The original 500 fixed shares configuration was not market-neutral. JPM at $220 × 500 shares = $110,000 versus BAC at $45 × 500 shares = $22,500. True dollar-neutral sizing removes the directional JPM exposure that inflated early backtest results.
Best configuration (in-sample): +1.82% return, Sharpe 1.99, profit factor 2.20, max drawdown -1.02%, win rate 60% on the current 60-day 5-minute window with dollar-neutral $50k/leg sizing, Z=1.5, rolling cointegration gate, and 1% stop loss.
Honest limitation: The best result was produced on in-sample data. The pair was selected after observing performance, which introduces selection bias. Out-of-sample validation via IBKR paper trading is ongoing.

Installation
bashpip install yfinance pandas numpy matplotlib statsmodels scipy ib_insync
For backtesting only (no IBKR required):
bashpython backtest_nsai.py        # run backtest on current 60-day window
python screeningpairs.py       # scan for cointegrated pairs
python coint_history.py        # plot rolling cointegration history for JPM/BAC
For paper trading (requires IBKR IB Gateway):

Open IB Gateway, log into paper trading account
Configure → Settings → API: port 7497, Read-Only API unchecked
python paper_trading.py


Configuration
All parameters are at the top of backtest_nsai.py:
ParameterValueDescriptionZ_THRESHOLD1.5Entry signal thresholdDOLLAR_SIZE50000Notional per legSTOP_LOSS_PCT0.01Max loss per trade (1% of notional)ROLLING_COINT_BARS500Bars for rolling cointegration gateROLLING_COINT_PVALUE0.10Gate significance thresholdWARMUP_BARS100Bars before trading begins
Exit threshold (abs(z) < 0.1) and timeout (50 bars) are in nsaibrain.py.

Files
nsaibrain.py          Core engine — Kalman filter, Hurst, signal generation
backtest_nsai.py      Backtesting framework with dashboard visualization
screeningpairs.py     Pair scanner across 80+ tickers / 13 sectors
coint_history.py      Rolling cointegration history analyzer
paper_trading.py      Live paper trading via IBKR IB Gateway

Limitations and Future Work

Strategy validated on one primary pair (JPM/BAC). A robust strategy needs 10–20 pairs.
5-minute yfinance data only covers the last 60 days, limiting out-of-sample testing.
The Kalman filter assumes linear Gaussian dynamics. A particle filter or neural Kalman filter could better handle the non-Gaussian spread behavior during high-volatility regimes.
Replacing the Z-score threshold with a gradient boosted tree model trained on trade outcomes (Z-score, Hurst, rolling p-value, time of day, volume ratio) is a natural extension.
