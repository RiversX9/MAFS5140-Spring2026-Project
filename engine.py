import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class BacktestEngine:
    def __init__(self, data_feed, strategy):
        self.data_feed = data_feed
        self.strategy = strategy

    def run(self) -> pd.Series:
        """
        Executes the backtest loop.
        Returns a Pandas Series of the portfolio's periodic returns.
        """
        portfolio_returns = {}
        prev_prices = None
        prev_weights = None

        total_steps = None
        if hasattr(self.data_feed, "data"):
            try:
                total_steps = len(self.data_feed.data)
            except TypeError:
                total_steps = None

        use_tqdm = tqdm is not None
        if use_tqdm:
            iterator = tqdm(self.data_feed, total=total_steps, desc="Backtest", unit="step")
        else:
            iterator = self.data_feed

        print("Starting backtest...")

        for step_idx, (timestamp, current_prices) in enumerate(iterator, start=1):
            if not use_tqdm and step_idx % 100 == 0:
                if total_steps:
                    pct = (step_idx / total_steps) * 100
                    print(f"Backtest progress: {step_idx}/{total_steps} ({pct:5.1f}%)", end="\r", flush=True)
                else:
                    print(f"Backtest progress: {step_idx} steps", end="\r", flush=True)

            # 1. Calculate Portfolio Return for the current period
            if prev_prices is not None and prev_weights is not None:
                # Calculate percentage change of each asset
                asset_returns = (current_prices - prev_prices) / prev_prices
                
                # Portfolio return is the dot product of previous weights and current asset returns.
                # Since cash earns 0%, any weight < 1.0 naturally leaves the remainder as cash.
                port_return = (prev_weights * asset_returns).sum()
                portfolio_returns[timestamp] = port_return

            # 2. Get new weights from the student's strategy
            try:
                new_weights = self.strategy.step(current_prices)
            except Exception as e:
                raise RuntimeError(f"Strategy execution failed at {timestamp}.\nError: {e}")

            # 3. Validate the output
            self._validate_weights(new_weights, timestamp, current_prices.index)

            # 4. Update state for the next timestamp
            prev_prices = current_prices
            prev_weights = new_weights

        if not use_tqdm:
            print()

        print("Backtest completed successfully.")
        return pd.Series(portfolio_returns, name="Portfolio_Return")

    def _validate_weights(self, weights, timestamp, expected_tickers):
        """
        Strictly enforces the rules: output must be a Series, no shorting, no leverage.
        """
        if not isinstance(weights, pd.Series):
            raise TypeError(
                f"Validation Error at {timestamp}: Strategy must return a pandas Series. "
                f"Got {type(weights)} instead."
            )

        if not weights.index.equals(expected_tickers):
            raise ValueError(
                f"Validation Error at {timestamp}: The index of the returned weights "
                f"does not match the input tickers."
            )

        if (weights < 0).any():
            violators = weights[weights < 0].to_dict()
            raise ValueError(
                f"Validation Error at {timestamp}: Negative weights (short selling) are not allowed.\n"
                f"Violating assets: {violators}"
            )

        # Use a small epsilon (1e-6) to account for floating-point arithmetic inaccuracies
        weight_sum = weights.sum()
        if weight_sum > 1.000001:
            raise ValueError(
                f"Validation Error at {timestamp}: Sum of weights exceeds 1.0 (leverage is not allowed).\n"
                f"Total weight sum: {weight_sum:.4f}"
            )