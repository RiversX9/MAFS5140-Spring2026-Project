import pandas as pd
import numpy as np

class Evaluator:
    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        """
        periods_per_year: 252 for daily data, 252*390 for minute data, etc.
        """
        self.returns = returns
        self.periods_per_year = periods_per_year

    def cumulative_return(self) -> float:
        return (1 + self.returns).prod() - 1.0

    def annualized_return(self) -> float:
        cum_ret = self.cumulative_return()
        num_periods = len(self.returns)
        if num_periods == 0:
            return 0.0
        return (1 + cum_ret) ** (self.periods_per_year / num_periods) - 1.0

    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        ann_ret = self.annualized_return()
        ann_vol = self.annualized_volatility()
        if ann_vol == 0:
            return 0.0
        return (ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        cumulative_wealth = (1 + self.returns).cumprod()
        rolling_max = cumulative_wealth.cummax()
        drawdowns = (cumulative_wealth - rolling_max) / rolling_max
        return drawdowns.min()

    def generate_report(self):
        """
        Prints and returns a dictionary of all computed metrics.
        """
        metrics = {
            "Cumulative Return": f"{self.cumulative_return() * 100:.2f}%",
            "Annualized Return": f"{self.annualized_return() * 100:.2f}%",
            "Annualized Volatility": f"{self.annualized_volatility() * 100:.2f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio():.2f}",
            "Max Drawdown": f"{self.max_drawdown() * 100:.2f}%"
        }
        
        print("\n--- Strategy Performance Report ---")
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        print("-----------------------------------")
        
        return metrics