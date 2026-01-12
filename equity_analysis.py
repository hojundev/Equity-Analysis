import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
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
except Exception as e;
    print(f"Error downloading stock/index data: {e}")
    exit()

monthly_returns = np.log(data / data.shift(1))
monthly_returns = monthly_returns.dropna()

if monthly_returns.empty:
    print("Not enough data to calculate monthly returns (need at least 2 months of price data).")
    exit()

try:
    fred_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    fred_end_date = pd.to_datetime + pd.DateOffset(months=1)
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