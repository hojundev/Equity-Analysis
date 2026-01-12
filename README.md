# Quantitative Equity Analysis Tool

A financial modeling engine built in Python to evaluate the risk-adjusted returns of large-cap tech stocks against the S&P 500 benchmark. This tool automates the retrieval of financial data, calculates key performance metrics, and quantifies volatility to aid in investment decision-making.

## 📌 Overview

This project focuses on quantitative investment analysis by leveraging the **Yahoo Finance API** for historical price data and the **Federal Reserve Economic Data (FRED)** for risk-free rates. The core objective is to calculate the **Sharpe Ratio**, allowing for a direct comparison between individual stock performance (e.g., AAPL) and a market benchmark (S&P 500).

## 🚀 Key Features

* **Automated Data Pipeline:** Fetches monthly historical closing prices for user-defined tickers and benchmarks (default: AAPL vs. S&P 500).
* **Dynamic Risk-Free Rate:** Integrates the 3-Month Treasury Constant Maturity Rate (`TB3MS`) directly from FRED to calculate accurate excess returns.
* **Robust Data Alignment:** Uses date buffering and forward-filling (`ffill`) to ensure seamless alignment between monthly stock returns and economic data reporting dates.
* **Performance Metrics:** Calculates and prints:
    * Average Monthly & Annualized Returns (Logarithmic)
    * Monthly & Annualized Volatility (Standard Deviation using Square Root of Time rule)
    * Sharpe Ratio (Risk-Adjusted Return)

## 🛠️ Technologies Used

* **Python 3.x**
* **Pandas:** For time-series manipulation and data alignment (`reindex`, `DateOffset`).
* **NumPy:** For mathematical operations (log returns, square roots).
* **yfinance:** For retrieving stock market data.
* **pandas_datareader:** For retrieving economic data (FRED).

## 📊 Methodology

1.  **Data Ingestion:** Downloads monthly historical data for specified tickers.
2.  **Return Calculation:** Computes **Logarithmic Returns** to ensure time-additivity:
    $$r_t = \ln(\frac{P_t}{P_{t-1}})$$
3.  **Risk-Free Adjustment:** Fetches `TB3MS` data from FRED, aligns it to the stock data timeline, and calculates excess returns.
4.  **Annualization:** Converts monthly metrics to annualized figures:
    * $Return_{ann} = Return_{monthly} \times 12$
    * $Volatility_{ann} = Volatility_{monthly} \times \sqrt{12}$
5.  **Sharpe Ratio:**
    $$\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}$$

## 💻 Usage

### Prerequisites
Ensure you have the required libraries installed:
```bash
pip install pandas numpy yfinance pandas_datareader
