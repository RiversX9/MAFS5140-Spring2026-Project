import pandas as pd

class DataFeed:
    def __init__(self, file_path: str):
        """
        Loads the dataset from a parquet file.
        Expects a DataFrame with timestamps as the index and tickers as columns.
        """
        try:
            self.data = pd.read_parquet(file_path)
        except Exception as e:
            raise IOError(f"Failed to load data from {file_path}. Error: {e}")
            
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("The index of the dataset must be a DatetimeIndex.")
            
        if self.data.isna().any().any():
            raise ValueError("Dataset contains NaN values. Please clean the data before backtesting.")

    def __iter__(self):
        """
        Allows the engine to iterate over the data row-by-row.
        Yields a tuple of (timestamp, pd.Series of prices).
        """
        return self.data.iterrows()