
import yfinance as yf
import numpy as np

def get_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    
    if len(tickers) == 1:
        return data  # single ticker returns normal format
    
    # For multiple tickers: extract 'Adj Close' from the multi-index
    adj_close = data.xs('Adj Close', axis=1, level=1)
    return adj_close

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

from scipy.optimize import minimize

def optimize_portfolio(mean_returns, cov_matrix, risk_level, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    bounds = risk_tolerance_bounds(risk_level, num_assets)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_guess = num_assets * [1. / num_assets]

    result = minimize(lambda x: -portfolio_stats(x, mean_returns, cov_matrix, risk_free_rate)[2],
                      init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result
