import pandas as pd
import numpy as np
from scipy.optimize import minimize

class Strategy:
    def __init__(self):
        self.lookback = 20
        self.top_n = 40
        self.rebalance_freq = 39
        self.w_momentum = 0.4
        self.w_lowvol = 0.35
        self.w_volume = 0.25

        self.price_history = pd.DataFrame()
        self.volume_history = pd.DataFrame()
        self.bar_counter = 0
        self.last_rebalance_bar = 0
        self.current_weights = None

    def step(self, current_market_data):
        self.bar_counter += 1
        current_prices = current_market_data['close']
        current_volumes = current_market_data['volume']

        self.price_history = pd.concat([self.price_history, current_prices.to_frame().T], ignore_index=True)
        self.volume_history = pd.concat([self.volume_history, current_volumes.to_frame().T], ignore_index=True)

        max_len = self.lookback + 1
        if len(self.price_history) > max_len:
            self.price_history = self.price_history.iloc[-max_len:]
            self.volume_history = self.volume_history.iloc[-max_len:]

        need_rebalance = (self.current_weights is None) or (self.bar_counter - self.last_rebalance_bar >= self.rebalance_freq)

        if need_rebalance and len(self.price_history) >= self.lookback + 1:
            self.current_weights = self._rebalance()
            self.last_rebalance_bar = self.bar_counter
        elif self.current_weights is None:
            n = len(current_prices)
            self.current_weights = pd.Series(1.0 / n, index=current_prices.index)
            self.last_rebalance_bar = self.bar_counter

        return self.current_weights

    def _rebalance(self):
        prices = self.price_history.iloc[-self.lookback-1:]
        volumes = self.volume_history.iloc[-self.lookback-1:]

        ret = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        pct_chg = prices.pct_change().dropna()
        vol = pct_chg.std()
        avg_vol = volumes.iloc[:-1].mean()
        vol_ratio = volumes.iloc[-1] / avg_vol

        def safe_normalize(series):
            if series.max() > series.min():
                return (series - series.min()) / (series.max() - series.min())
            return pd.Series(0.5, index=series.index)

        ret_norm = safe_normalize(ret)
        vol_norm = 1 - safe_normalize(vol)
        vol_ratio_norm = safe_normalize(vol_ratio)

        score = (self.w_momentum * ret_norm + self.w_lowvol * vol_norm + self.w_volume * vol_ratio_norm)
        selected = score.nlargest(self.top_n).index

        if len(selected) == 0:
            return pd.Series(0.0, index=score.index)

        selected_returns = pct_chg[selected].dropna()
        if len(selected_returns) < 2:
            weights = pd.Series(0.0, index=score.index)
            weights[selected] = 1.0 / len(selected)
            return weights

        # 关键修复：强制转换为 numpy 数组并确保可写
        returns_array = selected_returns.values
        Sigma = np.cov(returns_array, rowvar=False)
        Sigma = Sigma + np.eye(Sigma.shape[0]) * 1e-8
        Sigma = np.ascontiguousarray(Sigma)  # 确保是连续可写数组

        n = len(selected)

        def objective(w):
            return w @ Sigma @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n)]
        w0 = np.ones(n) / n

        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
        else:
            optimal_weights = w0

        optimal_weights = optimal_weights / optimal_weights.sum()

        weights = pd.Series(0.0, index=score.index)
        weights[selected] = optimal_weights
        return weights
