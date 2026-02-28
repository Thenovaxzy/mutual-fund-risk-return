import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_metrics(returns_df, benchmark_ticker="^GSPC", risk_free_rate=0.04):
    """
    Calculates annualized return, annualized volatility, and beta for each fund.
    
    Args:
        returns_df (pd.DataFrame): Daily returns dataframe (from fetch_data)
        benchmark_ticker (str): The column name representing the benchmark
        risk_free_rate (float): Annual risk-free rate (e.g., 0.04 for 4%)
        
    Returns:
        pd.DataFrame: A dataframe containing the calculated metrics for each fund.
    """
    if benchmark_ticker not in returns_df.columns:
        raise ValueError(f"Benchmark ticker '{benchmark_ticker}' not found in returns data.")
        
    funds = [c for c in returns_df.columns if c != benchmark_ticker]
    
    # Daily risk free rate approximation
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    metrics = []
    
    benchmark_returns = returns_df[benchmark_ticker].values.reshape(-1, 1)
    # Excess benchmark returns
    excess_benchmark = benchmark_returns - daily_rf
    
    for fund in funds:
        fund_returns = returns_df[fund]
        
        # 1. Annualized Return
        # (1 + mean daily return) ^ 252 - 1 is more accurate than mean * 252 depending on compounding
        compounded_return = (1 + fund_returns).prod() ** (252 / len(fund_returns)) - 1
        
        # 2. Annualized Volatility (Risk)
        volatility = fund_returns.std() * np.sqrt(252)
        
        # 3. Beta (Linear Regression of excess returns over market excess returns)
        excess_fund = fund_returns.values.reshape(-1, 1) - daily_rf
        model = LinearRegression()
        model.fit(excess_benchmark, excess_fund)
        beta = model.coef_[0][0]
        
        # 4. Alpha (Annualized)
        alpha_daily = model.intercept_[0]
        alpha_annualized = alpha_daily * 252
        
        # 5. Sharpe Ratio
        sharpe = (compounded_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        metrics.append({
            "Fund": fund,
            "Return (Annualized)": compounded_return,
            "Volatility (Annualized)": volatility,
            "Beta": beta,
            "Alpha (Annualized)": alpha_annualized,
            "Sharpe Ratio": sharpe
        })
        
    return pd.DataFrame(metrics)

def perform_cross_sectional_regression(metrics_df):
    """
    Regresses Annualized Returns against Beta to find the empirical Security Market Line (SML).
    """
    X = metrics_df[['Beta']].values
    y = metrics_df['Return (Annualized)'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # y = mx + c  (Return = risk_premium * Beta + risk_free_rate_implied)
    risk_premium = model.coef_[0][0]
    intercept = model.intercept_[0]
    r_squared = model.score(X, y)
    
    return {
        "slope": risk_premium,
        "intercept": intercept,
        "r_squared": r_squared,
        "model": model
    }

if __name__ == "__main__":
    # Simple test with dummy data
    print("Testing metrics calculation...")
    dates = pd.date_range(start='1/1/2020', periods=10)
    data = {
        'FundA': [0.01, -0.005, 0.02, 0.01, -0.01, 0.03, -0.02, 0.01, 0.005, 0.015],
        'FundB': [0.005, -0.002, 0.01, 0.005, -0.005, 0.015, -0.01, 0.005, 0.002, 0.008],
        '^GSPC': [0.012, -0.008, 0.022, 0.011, -0.015, 0.035, -0.025, 0.012, 0.006, 0.018]
    }
    df = pd.DataFrame(data, index=dates)
    
    metrics = calculate_metrics(df)
    print("\nMetrics:")
    print(metrics)
    
    cs_reg = perform_cross_sectional_regression(metrics)
    print("\nCross-Sectional Regression (SML):")
    print(f"Slope (Market Risk Premium): {cs_reg['slope']}")
    print(f"Intercept (Implied Risk-Free Rate): {cs_reg['intercept']}")
    print(f"R-squared: {cs_reg['r_squared']}")
