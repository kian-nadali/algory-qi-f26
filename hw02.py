import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def present_value(cash_flows, rate):
    total = 0

    for t, cash_flow in enumerate(cash_flows, start=1):
        total += cash_flow / (1 + rate) ** t

    return total
    
print(present_value([10, 15, 20], 0.10))
def bond_price(face, coupon_rate, years, market_rate):
    coupon = face * coupon_rate
    cash_flows = [coupon] * years
    cash_flows[-1] += face

    return present_value(cash_flows, market_rate)

print(f"{bond_price(1000, 0.04, 10, 0.04):.2f}")

price_2 = bond_price(1000, 0.04, 10, 0.02)
price_4 = bond_price(1000, 0.04, 10, 0.04)
price_496 = bond_price(1000, 0.04, 10, 0.0496)

print("\nBond Prices")
print(f"Market rate 2.00%: ${price_2:.2f}")
print(f"Market rate 4.00%: ${price_4:.2f}")
print(f"Market rate 4.96%: ${price_496:.2f}")

rates = np.linspace(0, 0.10, 101)
prices = []
for rate in rates:
    price = bond_price(1000, 0.04, 10, rate)
    prices.append(price)

plt.plot(rates * 100, prices)
plt.xlabel("Market Rate (%)")
plt.ylabel("Bond Price ($)")
plt.title("10-Year 4% Coupon Bond: Price vs. Market Rate")
plt.grid(True)
plt.show()


tickers = ["AAPL", "JPM", "XOM"]
market = "SPY"
stock_prices = {}
for ticker in tickers:
    data = yf.Ticker(ticker).history(period="5y")
    stock_prices[ticker] = data["Close"].dropna()
    print(ticker, "trading days:", len(stock_prices[ticker]))

spy_data = yf.Ticker(market).history(period="5y")
spy_close = spy_data["Close"].dropna()
print("SPY trading days:", len(spy_close))

spy_returns = spy_close.pct_change().dropna()
results = []
for ticker in tickers:
    prices = stock_prices[ticker]
    annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (1 / 5) - 1
    stock_returns = prices.pct_change().dropna()
    annual_volatility = stock_returns.std() * np.sqrt(252)
    combined = pd.concat([stock_returns, spy_returns], axis=1).dropna()
    combined.columns = ["Stock", "SPY"]
    covariance = combined["Stock"].cov(combined["SPY"])
    market_variance = combined["SPY"].var()
    beta = covariance / market_variance
    results.append({
        "Ticker": ticker,
        "Return": annual_return,
        "Volatility": annual_volatility,
        "Beta": beta
    })

results_df = pd.DataFrame(results)

print("\nStock Results")
print(results_df)

spy_beta = spy_returns.cov(spy_returns) / spy_returns.var()
print(f"\nSPY beta against itself: {spy_beta:.2f}")

beta_ranking = results_df.sort_values("Beta", ascending=False)
print("\nRanking by Beta:")
print(beta_ranking[["Ticker", "Beta"]])

volatility_ranking = results_df.sort_values("Volatility", ascending=False)
print("\nRanking by Volatility:")
print(volatility_ranking[["Ticker", "Volatility"]])

