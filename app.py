
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_utils import get_price_data, calculate_returns, portfolio_stats, generate_random_portfolios, optimize_portfolio

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Stock Portfolio Optimizer using Modern Portfolio Theory")

tickers_input = st.text_area("Enter up to 20 stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()][:20]

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
risk_tolerance = st.selectbox("Select Risk Tolerance", ["Low", "Medium", "High"])

if st.button("Optimize Portfolio"):
    with st.spinner("Downloading data and optimizing..."):
        prices = get_price_data(tickers, start_date, end_date)
        returns = calculate_returns(prices)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        result = optimize_portfolio(mean_returns, cov_matrix, risk_tolerance)
        opt_weights = result.x
        ret, vol, sharpe = portfolio_stats(opt_weights, mean_returns, cov_matrix)

        st.subheader("Optimized Portfolio")
        st.write(f"Expected Annual Return: **{ret:.2%}**")
        st.write(f"Annual Volatility: **{vol:.2%}**")
        st.write(f"Sharpe Ratio: **{sharpe:.2f}**")

        weight_df = pd.DataFrame({'Ticker': tickers, 'Weight': np.round(opt_weights, 4)})
        st.dataframe(weight_df)

        fig, ax = plt.subplots()
        ax.pie(opt_weights, labels=tickers, autopct='%1.1f%%')
        ax.set_title('Portfolio Allocation')
        st.pyplot(fig)

        results = generate_random_portfolios(5000, mean_returns, cov_matrix, 0.01)
        results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe', 'Weights'])

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis')
        ax2.scatter(vol, ret, c='red', s=50, label='Optimal Portfolio')
        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Expected Return')
        ax2.set_title('Efficient Frontier')
        ax2.legend()
        st.pyplot(fig2)
