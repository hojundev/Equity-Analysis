import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

tickers = []
risk_free_ticker = 'TB3MS'
start_date = ''
end_date = ''

def metric_calculations(data, rf):
    sharpe_ratios = {}
    for ticker in tickers:
        if ticker not in data.columns:
            print(f"Ticker {ticker} not found in downloaded data columns.")
            continue
        asset_returns = data[ticker].dropna()
        if len(asset_returns) < 2:
            print(f"Not enough return data points for {ticker} to calculate Sharpe Ratio.")
            sharpe_ratios[ticker] = np.nan
            continue
        avg_return = asset_returns.mean()
        annualized_return = (1 + avg_return)**12 - 1
        std_monthly = asset_returns.std()
        annualized_volatility = std_monthly * np.sqrt(12)
        if annualized_volatility == 0:
            print(f"Volatility for {ticker} is zero, cannot calculate Sharpe Ratio.")
            sharpe_ratios[ticker] = np.nan
        else :
            sharpe_ratio = (annualized_return - rf) / annualized_volatility
        sharpe_ratios[ticker] = sharpe_ratio

        print(f"\nTicker: {ticker}")
        print(f"  Average Monthly Return: {avg_return:.4%}")
        print(f"  Annualized Return: {annualized_return:.4%}")
        print(f"  Monthly Volatility (Std Dev): {std_monthly:.4%}")
        print(f"  Annualized Volatility: {annualized_volatility:.4%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")

    return sharpe_ratios
        

def comparison_analysis(data):
    log_returns = np.log(data / data.shift(1))
    log_returns.dropna(inplace=True)

    if log_returns.empty:
        print("Not enough data to calculate monthly returns")
        return

    if not isinstance(log_returns, pd.DataFrame):
        log_returns = pd.DataFrame(log_returns, columns=tickers)
    
    # risk free rate
    fred_start_date = pd.to_datetime(start_date) - pd.DateOffset(months=1)
    fred_end_date = pd.to_datetime(end_date) + pd.DateOffset(months=1)
    rf_data = pdr.get_data_fred(risk_free_ticker, start = fred_start_date, end = fred_end_date)
    if rf_data.empty:
        print("No risk free rate data found.")
        return
    rf_data[risk_free_ticker] = rf_data[risk_free_ticker] / 100.0
    rf_data = rf_data.reindex(log_returns.index, method='ffill').dropna()
    if rf_data.empty:
        print("Not enough risk free rate data.")
        return
    average_annualized_rf = rf_data[risk_free_ticker].mean()

    sharpe_ratios = metric_calculations(log_returns, average_annualized_rf)

    # --- Plotting ---
    sns.set_style("whitegrid")
    plt.style.use('ggplot')

    cumulative_returns = np.exp(log_returns.cumsum())

    perf_summary = {
        'Ticker': [],
        'Ann_Return': [],
        'Ann_Vol': [],
        'Sharpe': []
    }

    for ticker in tickers:
        if ticker in sharpe_ratios and not np.isnan(sharpe_ratios[ticker]):
            asset_ret = log_returns[ticker].dropna()
            ann_ret = asset_ret.mean() * 12
            ann_vol = asset_ret.std() * np.sqrt(12)
            
            perf_summary['Ticker'].append(ticker)
            perf_summary['Ann_Return'].append(ann_ret)
            perf_summary['Ann_Vol'].append(ann_vol)
            perf_summary['Sharpe'].append(sharpe_ratios[ticker])

    perf_df = pd.DataFrame(perf_summary)

    fig = plt.figure(figsize=(14, 10))
    plt.suptitle(f'Equity Comparison Dashboard\n({start_date} to {end_date})', fontsize=16, weight='bold')

    cum_return = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    for col in cumulative_returns.columns:
        cum_return.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
    cum_return.set_title('Growth of $1 Investment (Cumulative)', fontsize=12)
    cum_return.set_ylabel('Growth Factor')
    cum_return.legend(loc='upper left')
    cum_return.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)

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


def main():
    global tickers, start_date, end_date
    tickers = input("Enter the tickers separated by commas: ").split(',')
    for i in range(len(tickers)): 
        tickers[i] = tickers[i].strip().upper()
    
    start_date = input("Enter the start date in the format MM/DD/YYYY: ")
    start_date = start_date[6:] + "-" + start_date[0:2] + "-" + start_date[3:5]

    end_date = input("Enter the end date in the format MM/DD/YYYY: ")
    end_date = end_date[6:] + "-" + end_date[0:2] + "-" + end_date[3:5]

    data = yf.download(tickers, start=start_date, end=end_date, interval="1mo")['Close']
    data.dropna(inplace=True)
    if data.empty:
        print("No data found for the given tickers and date range.")
        return

    comparison_analysis(data)

if __name__ == "__main__":
    main()