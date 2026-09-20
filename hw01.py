import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

ticker = "DKS"
market = "SPY"

dks = yf.Ticker(ticker).history(period="1y")
spy = yf.Ticker(market).history(period="1y")

dks_close = dks["Close"].dropna()
spy_close = spy["Close"].dropna()

print(ticker, "trading days:", len(dks_close))
print("First date:", dks_close.index[0].date())
print("Last date:", dks_close.index[-1].date())

print(market, "trading days:", len(spy_close))
print("First date:", spy_close.index[0].date())
print("Last date:", spy_close.index[-1].date())

dks_return = (dks_close.iloc[-1] / dks_close.iloc[0]) - 1
spy_return = (spy_close.iloc[-1] / spy_close.iloc[0]) - 1
dks_daily_returns = dks_close.pct_change().dropna()
spy_daily_returns = spy_close.pct_change().dropna()
dks_volatility = dks_daily_returns.std() * np.sqrt(252)
spy_volatility = spy_daily_returns.std() * np.sqrt(252)

print("\nDKS Results")
print(f"Last close: ${dks_close.iloc[-1]:.2f}")
print(f"Year return: {dks_return:.2%}")
print(f"Annualized volatility: {dks_volatility:.2%}")

print("\nSPY Results")
print(f"Last close: ${spy_close.iloc[-1]:.2f}")
print(f"Year return: {spy_return:.2%}")
print(f"Annualized volatility: {spy_volatility:.2%}")

dks_rebased = dks_close / dks_close.iloc[0] * 100
spy_rebased = spy_close / spy_close.iloc[0] * 100

plt.plot(dks_rebased.index, dks_rebased, label="DKS")
plt.plot(spy_rebased.index, spy_rebased, label="SPY")

plt.xlabel("Date")
plt.ylabel("Rebased Price (Starting Value = 100)")
plt.title("DKS vs. SPY - One Year Performance")
plt.legend()

plt.show()

biggest_move_date = dks_daily_returns.abs().idxmax()
biggest_move = dks_daily_returns.loc[biggest_move_date]

print("\nDKS Biggest Single-Day Move")
print("Date:", biggest_move_date.date())
print(f"Move: {biggest_move:.2%}")


