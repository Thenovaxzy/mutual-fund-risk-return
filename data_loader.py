import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, benchmark_ticker="^GSPC", years=5):
    """
    Fetches historical adjusted close prices for the given tickers and a benchmark.
    
    Args:
        tickers (list or str): List of mutual fund/stock ticker symbols or a single string comma-separated
        benchmark_ticker (str): Ticker for the benchmark index (default is S&P 500: ^GSPC)
        years (int): Number of years of historical data to fetch
    
    Returns:
        pd.DataFrame: DataFrame containing daily returns for all tickers and the benchmark.
    """
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(',')]
        
    all_tickers = tickers + [benchmark_ticker]
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    
    # Download data
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    # Use threads=False to avoid some occasional yfinance issues, though it's slower
    data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)["Close"]
    
    if isinstance(data, pd.Series):
         data = pd.DataFrame({all_tickers[0]: data})
         
    # Handle potentially missing data
    data = data.dropna()
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    return returns

if __name__ == "__main__":
    # Test the function
    sample_tickers = ["VTSAX", "FXAIX", "PRASX"]
    print(f"Fetching data for: {sample_tickers}")
    try:
        ret_data = fetch_data(sample_tickers, years=1)
        print("Data successfully fetched!")
        print(ret_data.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
