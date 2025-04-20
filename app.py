
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squarify
from portfolio_utils import (
    get_price_data,
    calculate_returns,
    portfolio_stats,
    generate_random_portfolios,
    optimize_portfolio,
    validate_tickers,
    get_sector_mapping
)

st.set_page_config(page_title="Portfolio Optimizer Pro", layout="wide")
st.title("📈 Advanced Stock Portfolio Optimizer")

# User Inputs
tickers_input = st.text_area("Enter up to 20 stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()][:20]

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
risk_tolerance = st.selectbox("Select Risk Tolerance", ["Low", "Medium", "High"])
risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100

if st.button("Optimize Portfolio"):
    tickers = validate_tickers(tickers)
    if not tickers:
        st.error("No valid tickers found. Please check your input.")
        st.stop()

    prices = get_price_data(tickers, start_date, end_date)
    if prices.empty:
        st.error("No price data available. Try different tickers or date range.")
        st.stop()

    returns = calculate_returns(prices)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    result = optimize_portfolio(mean_returns, cov_matrix, risk_tolerance, risk_free_rate)
    opt_weights = result.x

    weight_df = pd.DataFrame({'Ticker': prices.columns, 'Weight': np.round(opt_weights, 4)})
    weight_df['Sector'] = weight_df['Ticker'].map(get_sector_mapping(prices.columns))
    non_zero_df = weight_df[weight_df['Weight'] > 0]

    selected_tickers = non_zero_df['Ticker'].tolist()
    prices = prices[selected_tickers]
    returns = calculate_returns(prices)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    st.subheader("📊 Optimized Portfolio Statistics")
    ret, vol, sharpe = portfolio_stats(opt_weights, mean_returns, cov_matrix, risk_free_rate)

    # Portfolio Analytics
    benchmark_data = yf.download(['SPY'], start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    benchmark_data = benchmark_data.xs('Close', axis=1, level=1)
    spy_returns = benchmark_data['SPY'].pct_change().dropna()

    portfolio_returns = returns @ opt_weights[:len(returns.columns)]
    opt_cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = opt_cum_returns.cummax()
    drawdown = opt_cum_returns / rolling_max - 1
    max_drawdown = drawdown.min()
    VaR = np.percentile(portfolio_returns * -1, 5)
    CVaR = (portfolio_returns[portfolio_returns < -VaR]).mean() * -1
    beta = np.cov(portfolio_returns, spy_returns)[0, 1] / np.var(spy_returns)
    cumulative_return = (opt_cum_returns.iloc[-1] - 1)

    st.markdown(f'''
- **Expected Annual Return:** {ret:.2%}  
- **Annual Volatility:** {vol:.2%}  
- **Sharpe Ratio:** {sharpe:.2f}  
- **Portfolio Beta (vs S&P 500):** {beta:.2f}  
- **Max Drawdown:** {max_drawdown:.2%}  
- **Value at Risk (95%):** {VaR:.2f}  
- **Conditional VaR (CVaR):** {CVaR:.2f}  
- **Cumulative Return (Backtest):** {cumulative_return:.2%}  
''')

    st.subheader("📦 Portfolio Allocation Treemap")
    available_sectors = sorted(non_zero_df['Sector'].unique())
    selected_sectors = st.multiselect("Filter sectors to show in treemap:", options=available_sectors, default=available_sectors)
    filtered_df = non_zero_df[non_zero_df['Sector'].isin(selected_sectors)]

    if not filtered_df.empty:
        sector_colors = dict(zip(filtered_df['Sector'].unique(), plt.cm.tab20.colors[:len(filtered_df['Sector'].unique())]))
        colors = [sector_colors[s] for s in filtered_df['Sector']]

        fig, ax = plt.subplots(figsize=(12, 7))
        squarify.plot(
            sizes=filtered_df['Weight'],
            label=[f"{t} ({w:.2%})" for t, w in zip(filtered_df['Ticker'], filtered_df['Weight'])],
            color=colors,
            alpha=0.8
        )
        plt.title("Portfolio Allocation Treemap (Grouped by Sector)")
        plt.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No stocks to display for the selected sectors.")

    st.subheader("📌 Correlation Matrix of Portfolio Assets")
    corr = returns[selected_tickers].corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)
