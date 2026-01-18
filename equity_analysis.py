import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

tickers = ['AAPL', '^GSPC']
start_date = '2023-01-01'
end_date = '2026-01-01'
risk_free_ticker = 'TB3MS'

today_str = datetime.today().strftime('%Y-%m-%d')
effective_end_date_yf = min(end_date, today_str)

try:
    data = yf.download(tickers, start=start_date, end=effective_end_date_yf, interval='1mo')['Close']
    if data.empty:
        print(f"No price data found for the given tickers and date range (up to {effective_end_date_yf}).")
        exit()
    data.dropna(how='all', inplace=True)
    if data.empty:
        print(f"Price data became empty after dropping NaNs. Check your date range and tickers.")
        exit()
except Exception as e:
    print(f"Error downloading stock/index data: {e}")
    exit()

monthly_returns = np.log(data / data.shift(1))
monthly_returns = monthly_returns.dropna()

if monthly_returns.empty:
    print("Not enough data to calculate monthly returns (need at least 2 months of price data).")
    exit()

try:
    fred_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    fred_end_date = pd.to_datetime(end_date) + pd.DateOffset(months=1)
    rf_data = pdr.get_data_fred(risk_free_ticker, start=fred_start_date, end=fred_end_date)
    if rf_data.empty:
        print(f"No risk-free rate data found for {risk_free_ticker}.")
        exit()
    rf_data[risk_free_ticker] = rf_data[risk_free_ticker]/ 100.0
    rf_data = rf_data.reindex(monthly_returns.index, method='ffill').dropna()
    if rf_data.empty:
        print(f"Could not align risk-free rate data with asset returns.")
        exit()
    average_annualized_rf = rf_data[risk_free_ticker].mean()
except Exception as e:
    print(f"Error downloading or processing risk-free rate data: {e}")
    exit()

sharpe_ratios = {}
annualization_factor_returns = 12
annualization_factor_volatility = np.sqrt(12)

if not isinstance(monthly_returns, pd.DataFrame):
    if len(tickers) == 1:
        monthly_returns = pd.DataFrame(monthly_returns, columns=tickers)
    else:
        print("Error: monthly_returns is not a DataFrame as expected.")
        exit()

for ticker in tickers:
    if ticker not in monthly_returns.columns:
        print(f"Warning: Ticker {ticker} not found in downloaded data columns.")
        continue
    asset_returns = monthly_returns[ticker].dropna()
    if len(asset_returns) < 2:
        print(f"Not enough return data points for {ticker} to calculate Sharpe Ratio.")
        sharpe_ratios[ticker] = np.nan
        continue
    avg_monthly_return = asset_returns.mean()
    annualized_return = avg_monthly_return * annualization_factor_returns
    std_dev_monthly = asset_returns.std()
    annualized_volatility = std_dev_monthly * annualization_factor_volatility
    if annualized_volatility == 0:
        sharpe_ratio = np.nan
        print(f"Volatility for {ticker} is zero, Sharpe Ratio cannot be calculated")
    else:
        sharpe_ratio = (annualized_return - average_annualized_rf) / annualized_volatility
    sharpe_ratios[ticker] = sharpe_ratio
    print(f"\nTicker: {ticker}")
    print(f"  Average Monthly Return: {avg_monthly_return:.4%}")
    print(f"  Annualized Return: {annualized_return:.4%}")
    print(f"  Monthly Volatility (Std Dev): {std_dev_monthly:.4%}")
    print(f"  Annualized Volatility: {annualized_volatility:.4%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")

print(f"\nNote: Calculations are based on monthly data from {monthly_returns.index.min().strftime('%Y-%m-%d')} to {monthly_returns.index.max().strftime('%Y-%m-%d')}.")
if effective_end_date_yf < end_date:
    print(f"The requested end date was {end_date}, but data was only available up to {effective_end_date_yf}.")

sns.set_style("whitegrid")
plt.style.use('ggplot')

cumulative_returns = np.exp(monthly_returns.cumsum())

perf_summary = {
    'Ticker': [],
    'Ann_Return': [],
    'Ann_Vol': [],
    'Sharpe': []
}

for ticker in tickers:
    if ticker in sharpe_ratios and not np.isnan(sharpe_ratios[ticker]):
        asset_ret = monthly_returns[ticker].dropna()
        ann_ret = asset_ret.mean() * 12
        ann_vol = asset_ret.std() * np.sqrt(12)
        
        perf_summary['Ticker'].append(ticker)
        perf_summary['Ann_Return'].append(ann_ret)
        perf_summary['Ann_Vol'].append(ann_vol)
        perf_summary['Sharpe'].append(sharpe_ratios[ticker])

perf_df = pd.DataFrame(perf_summary)

fig = plt.figure(figsize=(14, 10))
plt.suptitle(f'Equity Analysis Dashboard\n({start_date} to {effective_end_date_yf})', fontsize=16, weight='bold')

cum_return = plt.subplot2grid((2, 2), (0, 0), colspan=2)
for col in cumulative_returns.columns:
    cum_return.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
cum_return.set_title('Growth of $1 Investment (Cumulative)', fontsize=12)
cum_return.set_ylabel('Growth Factor')
cum_return.legend(loc='upper left')
cum_return.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5) # Break-even line

risk_return = plt.subplot2grid((2, 2), (1, 0))
risk_return.scatter(perf_df['Ann_Vol'], perf_df['Ann_Return'], s=100, alpha=0.7, c='royalblue')
for i, txt in enumerate(perf_df['Ticker']):
    risk_return.annotate(txt, (perf_df['Ann_Vol'][i], perf_df['Ann_Return'][i]), 
                 xytext=(5, 5), textcoords='offset points', weight='bold')
risk_return.set_title('Risk (Volatility) vs Return', fontsize=12)
risk_return.set_xlabel('Annualized Volatility (Std Dev)')
risk_return.set_ylabel('Annualized Return')
risk_return.grid(True, which='both', linestyle='--', alpha=0.7)

sharpe_ratio_compare = plt.subplot2grid((2, 2), (1, 1))
colors = ['green' if x > 0 else 'red' for x in perf_df['Sharpe']]
bars = sharpe_ratio_compare.bar(perf_df['Ticker'], perf_df['Sharpe'], color=colors, alpha=0.7)
sharpe_ratio_compare.set_title('Sharpe Ratio (Risk-Adjusted Return)', fontsize=12)
sharpe_ratio_compare.axhline(0, color='black', linewidth=1)
sharpe_ratio_compare.set_ylabel('Sharpe Ratio')
for bar in bars:
    height = bar.get_height()
    sharpe_ratio_compare.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()