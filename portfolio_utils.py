
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import pandas as pd

def get_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    if len(tickers) == 1:
        # Return single ticker as a single-column DataFrame
        return data.to_frame(name=tickers[0]) if isinstance(data, pd.Series) else data[['Close']].rename(columns={'Close': tickers[0]})
    
    # Multi-ticker case with MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        return data.xs('Close', axis=1, level=1)
    
    raise ValueError("Unexpected data structure returned from yfinance.")


def calculate_returns(prices):
    return prices.pct_change().dropna()

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    port_return = np.dot(weights, mean_returns) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe_ratio

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = []
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        ret, vol, sharpe = portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        results.append([ret, vol, sharpe, weights])
    return results

def risk_tolerance_bounds(risk_level, num_assets):
    if risk_level == "Low":
        return tuple((0.02, 0.1) for _ in range(num_assets))
    elif risk_level == "Medium":
        return tuple((0, 0.25) for _ in range(num_assets))
    else:
        return tuple((0, 1) for _ in range(num_assets))

def optimize_portfolio(mean_returns, cov_matrix, risk_level, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    bounds = risk_tolerance_bounds(risk_level, num_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = num_assets * [1. / num_assets]
    result = minimize(lambda x: -portfolio_stats(x, mean_returns, cov_matrix, risk_free_rate)[2],
                      init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def validate_tickers(tickers):
    valid = []
    for t in tickers:
        try:
            data = yf.Ticker(t).history(period="1d")
            if not data.empty:
                valid.append(t)
        except Exception:
            continue
    return valid

def get_sector_mapping(tickers):
    sector_map = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector", "Unknown")
        except Exception:
            sector_map[ticker] = "Unknown"
    return sector_map
