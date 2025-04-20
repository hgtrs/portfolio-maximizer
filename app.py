
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

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📈 Stock Portfolio Optimizer using Modern Portfolio Theory")

tickers_input = st.text_area("Enter up to 20 stock tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()][:20]

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
risk_tolerance = st.selectbox("Select Risk Tolerance", ["Low", "Medium", "High"])
risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100

if st.button("Optimize Portfolio"):
    with st.spinner("Optimizing your portfolio..."):
        tickers = validate_tickers(tickers)
        prices = get_price_data(tickers, start_date, end_date).dropna(axis=1)
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

        st.subheader("✅ Optimized Portfolio")
        ret, vol, sharpe = portfolio_stats(opt_weights, mean_returns, cov_matrix, risk_free_rate)
        st.write(f"Expected Annual Return: **{ret:.2%}**")
        st.write(f"Annual Volatility: **{vol:.2%}**")
        st.write(f"Sharpe Ratio: **{sharpe:.2f}**")
        st.dataframe(weight_df)

        st.markdown("### 🧠 Portfolio Insight")
        top_names = ', '.join(non_zero_df['Ticker'].head(3)) + ('...' if len(non_zero_df) > 3 else '')
        st.markdown(
            f"This portfolio includes stocks like **{top_names}**, selected for high return-to-risk ratios. "
            f"These assets help maximize the **Sharpe ratio**, reduce **volatility**, and improve overall performance."
        )

        st.markdown("### 🗂️ Sector Allocation Treemap")
        available_sectors = sorted(non_zero_df['Sector'].unique())
        selected_sectors = st.multiselect("Filter sectors to show in treemap:", options=available_sectors, default=available_sectors)
        filtered_df = non_zero_df[non_zero_df['Sector'].isin(selected_sectors)]

        if not filtered_df.empty:
            labels = [f"{row['Ticker']} ({row['Weight']:.2%})" for _, row in filtered_df.iterrows()]
            color_vals = filtered_df['Sector'].astype('category').cat.codes
            colors = plt.cm.tab20c(color_vals / (color_vals.max() + 1))

            fig, ax = plt.subplots(figsize=(12, 7))
            squarify.plot(sizes=filtered_df['Weight'], label=labels, color=colors, alpha=0.8)
            plt.title("Portfolio Allocation Treemap (Filtered by Sector)")
            plt.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No stocks to display for the selected sectors.")

        st.subheader("📊 Efficient Frontier")
        results = generate_random_portfolios(5000, mean_returns, cov_matrix, risk_free_rate)
        results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe', 'Weights'])

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe'], cmap='viridis')
        ax2.scatter(vol, ret, c='red', s=50, label='Optimal Portfolio')
        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Expected Return')
        ax2.set_title('Efficient Frontier')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("📈 Performance Comparison vs Benchmarks")
        benchmark_data = yf.download(['SPY', 'IWM'], start=start_date, end=end_date)['Adj Close']
        benchmark_returns = benchmark_data.pct_change().dropna()

        opt_cum_returns = (1 + returns @ opt_weights[:len(returns.columns)]).cumprod()
        spy_cum = (1 + benchmark_returns['SPY']).cumprod()
        iwm_cum = (1 + benchmark_returns['IWM']).cumprod()

        performance_df = pd.DataFrame({
            'Optimized Portfolio': opt_cum_returns,
            'S&P 500 (SPY)': spy_cum,
            'Russell 2000 (IWM)': iwm_cum
        })

        fig3, ax3 = plt.subplots()
        performance_df.plot(ax=ax3)
        ax3.set_title("Growth of $1 Over Time")
        ax3.set_ylabel("Portfolio Value")
        ax3.set_xlabel("Date")
        st.pyplot(fig3)

        st.markdown("### 📌 Performance Summary")
        final_returns = performance_df.iloc[-1] / performance_df.iloc[0] - 1
        vols = performance_df.pct_change().std() * np.sqrt(252)

        def summarize_perf(name):
            return f"- **{name}**: Return = {final_returns[name]:.2%}, Volatility = {vols[name]:.2%}"

        st.markdown(summarize_perf("Optimized Portfolio"))
        st.markdown(summarize_perf("S&P 500 (SPY)"))
        st.markdown(summarize_perf("Russell 2000 (IWM)"))

        st.markdown("### 💬 Key Takeaways")
        if final_returns['Optimized Portfolio'] > final_returns['S&P 500 (SPY)']:
            st.markdown("- ✅ Outperformed the S&P 500 overall.")
        else:
            st.markdown("- ⚠️ Underperformed the S&P 500 overall.")

        if vols['Optimized Portfolio'] < vols['S&P 500 (SPY)']:
            st.markdown("- ✅ Lower volatility than the S&P 500, suggesting better stability.")
        else:
            st.markdown("- ⚠️ Higher volatility than S&P 500, potentially due to concentrated bets.")

        if final_returns['Optimized Portfolio'] > final_returns['Russell 2000 (IWM)']:
            st.markdown("- ✅ Outperformed the Russell 2000 index over the selected period.")
        else:
            st.markdown("- ⚠️ Trailed the Russell 2000, possibly due to missing out on small-cap rallies.")
