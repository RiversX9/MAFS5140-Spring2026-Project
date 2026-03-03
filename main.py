from data_feed import DataFeed
from engine import BacktestEngine
from evaluator import Evaluator

# Import the student's strategy
# Ensure strategy.py is in the same directory
from strategy import Strategy 

def main():
    # 1. Define the path to the dataset
    # (Students will point this to the sample data you provide them)
    data_path = "data_downloader/test_close_price_data.parquet" 
    
    try:
        # 2. Initialize components
        print("Loading data...")
        feed = DataFeed(data_path)
        
        print("Initializing strategy...")
        student_strategy = Strategy()
        
        engine = BacktestEngine(data_feed=feed, strategy=student_strategy)
        
        # 3. Run the backtest
        portfolio_returns = engine.run()
        
        # 4. Evaluate the results
        # Note: Adjust periods_per_year based on your dataset 
        # (e.g., 252 for daily data, 65520 for 15-min data)
        evaluator = Evaluator(portfolio_returns, periods_per_year=252)
        evaluator.generate_report()

    except Exception as e:
        print(f"\n[BACKTEST FAILED] {type(e).__name__}: {e}")
        print("Please fix the error in your strategy and try again.")

if __name__ == "__main__":
    main()