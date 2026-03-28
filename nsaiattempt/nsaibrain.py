import numpy as np
from statsmodels.tsa.stattools import coint


class NSAIEngine:
    """
    NSAI Pairs Trading Engine — Best Configuration
    -----------------------------------------------
    Kalman filter tracks dynamic hedge ratio (beta) and intercept (alpha).
    Standardized Kalman innovation Z-score is the trading signal.
    Hurst exponent filters out non-mean-reverting regimes bar-by-bar.

    Best parameters found through research:
    - Z entry threshold: 1.5
    - Z exit threshold:  0.1 (let winners run to full reversion)
    - Position timeout:  50 bars (~4 hours at 5-minute bars)
    - OLS initialization: seeded from returns-based regression, not [0,0]
    - Dollar-neutral sizing: handled in backtest, not in engine
    """

    def __init__(self, share_size=500, round_trip_fee=10.0,
                 Q_diag=1e-4, R=0.01, P_init=1.0,
                 require_warmup=True,
                 initial_beta=1.0, initial_alpha=0.0):
        self.share_size     = share_size
        self.round_trip_fee = round_trip_fee
        self.require_warmup = require_warmup

        # Kalman state: x = [beta, alpha]
        # Seeded from OLS regression — not cold-started at [0, 0]
        self.x = np.array([initial_beta, initial_alpha])
        self.P = np.eye(2) * P_init
        self.Q = np.eye(2) * Q_diag
        self.R = R

        self.spread_history = []
        self.ticks_in_trade = 0
        self.is_warmed_up   = not require_warmup

    # ── OLS initialization helper ─────────────────────────────────────────────

    @staticmethod
    def ols_beta_alpha(p1_series, p2_series):
        """
        Compute hedge ratio and intercept from returns-based OLS regression.
        Uses log returns to avoid price-level bias (e.g. JPM=$220 vs BAC=$45
        would give beta≈5 on raw prices just from the price ratio).
        Returns (price_beta, alpha) for use as Kalman initial state.
        """
        p1 = np.asarray(p1_series, dtype=float)
        p2 = np.asarray(p2_series, dtype=float)
        r1 = np.diff(np.log(p1))
        r2 = np.diff(np.log(p2))
        returns_beta, _ = np.polyfit(r2, r1, 1)
        price_ratio  = p1.mean() / p2.mean()
        price_beta   = returns_beta * price_ratio
        alpha        = p1.mean() - price_beta * p2.mean()
        return float(price_beta), float(alpha)

    # ── Cointegration test ────────────────────────────────────────────────────

    @staticmethod
    def check_cointegration(p1_series, p2_series, pvalue_threshold=0.05):
        """
        Engle-Granger cointegration test.
        Returns dict: passed (bool), pvalue (float), message (str)
        """
        p1 = np.asarray(p1_series, dtype=float)
        p2 = np.asarray(p2_series, dtype=float)
        if len(p1) != len(p2):
            raise ValueError("Series must be the same length.")
        if len(p1) < 50:
            raise ValueError("Need at least 50 bars.")
        try:
            _, pvalue, _ = coint(p1, p2)
            passed  = pvalue < pvalue_threshold
            message = (f"PASS — cointegrated at p={pvalue:.4f}"
                       if passed else
                       f"FAIL — not cointegrated, p={pvalue:.4f}")
            return {"passed": passed, "pvalue": pvalue, "message": message}
        except Exception as e:
            return {"passed": False, "pvalue": 1.0,
                    "message": f"FAIL — error: {e}"}

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self, p1_series, p2_series):
        """
        Run Kalman in observe-only mode before trading begins.
        Because the engine is seeded with OLS beta/alpha, convergence
        is much faster than starting from [0, 0].
        Sets is_warmed_up=True when complete.
        """
        p1_series = np.asarray(p1_series, dtype=float)
        p2_series = np.asarray(p2_series, dtype=float)
        if len(p1_series) < 10:
            raise ValueError("Warmup series too short (minimum 10 bars).")
        for p1, p2 in zip(p1_series, p2_series):
            self.update_kalman(p1, p2)
            spread = p1 - (self.x[0] * p2 + self.x[1])
            self.spread_history.append(spread)
            if len(self.spread_history) > 50:
                self.spread_history.pop(0)
        self.is_warmed_up = True

    # ── Kalman filter ─────────────────────────────────────────────────────────

    def update_kalman(self, p1, p2):
        """One predict-update cycle. Returns beta, alpha, S, y."""
        self.P = self.P + self.Q
        H = np.array([[p2, 1.0]])
        y = p1 - float(H @ self.x)
        S = float(H @ self.P @ H.T) + self.R
        K = self.P @ H.T / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - K @ H) @ self.P
        return self.x[0], self.x[1], S, y

    # ── Hurst exponent ────────────────────────────────────────────────────────

    def get_hurst(self, ts):
        """
        H < 0.5  → mean-reverting (trade)
        H >= 0.5 → trending or random (wait)
        Returns 0.5 if insufficient data.
        """
        if len(ts) < 25:
            return 0.5
        lags = range(2, 20)
        tau  = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag])))
                for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]

    # ── Signal generation ─────────────────────────────────────────────────────

    def get_signal(self, p1, p2, current_pos=0, z_threshold=1.5):
        """
        Update Kalman and return trading signal.

        Entry: Hurst < 0.5 AND |z| > z_threshold (default 1.5)
        Exit:  |z| < 0.1 (full reversion — let winners run)
        Timeout: 50 bars (~4 hours at 5-minute bars)

        Returns dict: action, z, spread, beta
        """
        _, _, S, y = self.update_kalman(p1, p2)
        beta           = self.x[0]
        alpha          = self.x[1]
        current_spread = p1 - (beta * p2 + alpha)
        z_nsai         = y / np.sqrt(S)

        self.spread_history.append(current_spread)
        if len(self.spread_history) > 50:
            self.spread_history.pop(0)

        if not self.is_warmed_up:
            return {"action": "WAIT", "z": z_nsai,
                    "spread": current_spread, "beta": beta}

        h      = self.get_hurst(self.spread_history)
        action = "WAIT"

        if h < 0.5:
            if z_nsai > z_threshold and current_pos == 0:
                action = "SHORT"
            elif z_nsai < -z_threshold and current_pos == 0:
                action = "LONG"
            elif abs(z_nsai) < 0.1 and current_pos != 0:
                action = "EXIT"

        if current_pos != 0:
            self.ticks_in_trade += 1
            if self.ticks_in_trade > 50:
                action = "EXIT_TIMEOUT"
        else:
            self.ticks_in_trade = 0

        return {"action": action, "z": z_nsai,
                "spread": current_spread, "beta": beta}

    # ── State transfer ────────────────────────────────────────────────────────

    def get_state(self):
        return {
            "x":              self.x.copy(),
            "P":              self.P.copy(),
            "spread_history": list(self.spread_history),
            "is_warmed_up":   self.is_warmed_up,
        }

    def set_state(self, state):
        self.x              = state["x"].copy()
        self.P              = state["P"].copy()
        self.spread_history = list(state["spread_history"])
        self.is_warmed_up   = state.get("is_warmed_up", True)
