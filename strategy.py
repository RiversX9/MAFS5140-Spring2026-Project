import pandas as pd

class Strategy:
    def __init__(self):
        """
        Initialize any state variables here.
        This is called once at the very beginning of the backtest.
        """
        # We will use a list to store the historical price Series
        self.price_history = []
        self.lookback_period = 5

        """
        Core strategy logic. 
        This function is called at every timestamp by the BacktestEngine.
        
        INPUT:
        current_prices (pd.Series): Close prices of all assets at the current timestamp.
                                    Index = Tickers, Values = Prices.
                                    
        OUTPUT:
        pd.Series: Target weights for the portfolio.
                   Index = Tickers, Values = Weights (0.0 to 1.0).
                   Sum of weights must be <= 1.0.
        """
        # 1. Update internal state with the new data
        self.price_history.append(current_prices)
        
        # Keep only the required lookback period to save memory
        if len(self.price_history) > self.lookback_period:
            self.price_history.pop(0)
            
        # 2. Strategy Logic
        # If we don't have enough data yet, stay 100% in cash (return all zeros)
        if len(self.price_history) < self.lookback_period:
            return pd.Series(0.0, index=current_prices.index)
            
        # Convert our history list into a DataFrame to easily calculate the mean
        history_df = pd.DataFrame(self.price_history)
        moving_average = history_df.mean()
        
        # Identify assets where the current price is above its 5-period moving average
        bullish_assets = current_prices[current_prices > moving_average].index
        
        # 3. Portfolio Allocation
        # Initialize all weights to 0.0
        weights = pd.Series(0.0, index=current_prices.index)
        
        # Allocate equally among bullish assets
        num_bullish = len(bullish_assets)
        if num_bullish > 0:
            weight_per_asset = 1.0 / num_bullish
            weights[bullish_assets] = weight_per_asset
            
        # Return the weights. 
        # The engine will verify that weights >= 0 and weights.sum() <= 1.0
        return weights