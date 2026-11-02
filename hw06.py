import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timezone


# HW6 - Greeks, combinations, and implied volatility
# ---------------------------------------------------------- Q1
def payoff(prices, strike, premium, kind, position=1):
    if kind == 'call':
        values = np.maximum(prices - strike, 0) - premium
    else:
        values = np.maximum(strike - prices, 0) - premium
    return position * values

spot = 100
prices = np.linspace(0, 250, 2001)
strategies = {
    'Covered call': 100 * (prices - spot) + 100 * payoff(prices, 110, 3, 'call', -1),
    'Protective put': 100 * (prices - spot) + 100 * payoff(prices, 90, 2, 'put'),
    'Long straddle': 100 * payoff(prices, 100, 6, 'call') + 100 * payoff(prices, 100, 5, 'put'),
    'Bull call spread': 100 * payoff(prices, 95, 8, 'call') + 100 * payoff(prices, 110, 3, 'call', -1)
}


results = {
    'Covered call': ('', '$1,300', '$9,700', '97'),
    'Protective put': ('', 'unlimited', '$1,200', '102'),
    'Long straddle': ('', 'unlimited', '$1,100', '89, 111'),
    'Bull call spread': ('', '$1,000', '$500', '100')
}

for name, values in strategies.items():
    extra, max_profit, max_loss, breakeven = results[name]
    print(f'{name}: maximum profit = {max_profit}; maximum loss = {max_loss}; break-even = {breakeven}')

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, values) in zip(axes.flat, strategies.items()):
    ax.plot(prices, values)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel('Stock price at expiration ($)')
    ax.set_ylabel('Profit ($)')
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('hw06_combinations.png', dpi=160)
plt.show()

# ---------------------------------------------------------- Q3
def black_scholes(spot, strike, time, rate, volatility, kind):
    if time <= 0:
        if kind == 'call':
            return max(spot - strike, 0)
        return max(strike - spot, 0)
    d1 = (np.log(spot / strike) + (rate + volatility ** 2 / 2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    if kind == 'call':
        return spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    return strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)

call_price = black_scholes(100, 100, 1, 0.04, 0.25, 'call')
put_price = black_scholes(100, 100, 1, 0.04, 0.25, 'put')
print('\nBlack-Scholes call:', round(call_price, 2))
print('Black-Scholes put:', round(put_price, 2))
print('Put-call parity difference:', round(call_price - put_price, 2))

# ---------------------------------------------------------- Q4
def implied_volatility(market_price, spot, strike, time, rate, kind):
    low = 0.001
    high = 5.0
    if time <= 0 or market_price <= 0:
        return np.nan
    lower_price = black_scholes(spot, strike, time, rate, low, kind)
    upper_price = black_scholes(spot, strike, time, rate, high, kind)
    if market_price < lower_price - 1e-8 or market_price > upper_price + 1e-8:
        return np.nan
    for i in range(100):
        middle = (low + high) / 2
        guess = black_scholes(spot, strike, time, rate, middle, kind)
        if guess < market_price:
            low = middle
        else:
            high = middle
    return (low + high) / 2

print('Recovered volatility:', f'{implied_volatility(call_price, 100, 100, 1, 0.04, "call"):.4f}')

# ---------------------------------------------------------- Q5 and Q6
ticker = 'SPY'
rate = 0.04  # Assumed annual continuously compounded risk-free rate
stock = yf.Ticker(ticker)
history = stock.history(period='3mo', auto_adjust=True)
if len(history) < 31:
    raise RuntimeError('Not enough price history to calculate 30 daily returns.')
spot = float(history['Close'].iloc[-1])
returns = history['Close'].pct_change().dropna().tail(30)
realized_vol = float(returns.std(ddof=1) * np.sqrt(252))

expirations = stock.options
if not expirations:
    raise RuntimeError('No option expirations returned. Try again when Yahoo Finance is available.')
# Pick a reasonably near-dated expiry, ideally around one month out.
today = datetime.now(timezone.utc).date()
expiration = min(expirations, key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - today).days - 30))
time = (datetime.strptime(expiration, '%Y-%m-%d').date() - today).days / 365
if time <= 0:
    raise RuntimeError('Selected expiration is not in the future.')
chain = stock.option_chain(expiration).calls
strikes = []
volatilities = []
for i in range(len(chain)):
    bid = float(chain.iloc[i]['bid'])
    ask = float(chain.iloc[i]['ask'])
    strike = float(chain.iloc[i]['strike'])
    if bid > 0 and ask > 0 and ask >= bid:
        midpoint = (bid + ask) / 2
        vol = implied_volatility(midpoint, spot, strike, time, rate, 'call')
        if np.isfinite(vol):
            strikes.append(strike)
            volatilities.append(vol)

if not strikes:
    raise RuntimeError('No valid nonzero-bid calls available for IV calculation.')
order = np.argsort(strikes)
strikes = np.array(strikes)[order]
volatilities = np.array(volatilities)[order]
near = int(np.argmin(np.abs(strikes - spot)))
atm_vol = volatilities[near]

plt.figure(figsize=(9, 5))
plt.plot(strikes, volatilities, marker='o', markersize=3)
plt.axvline(spot, linestyle='--', label='Spot')
plt.xlabel('Strike ($)')
plt.ylabel('Implied volatility')
plt.title(f'{ticker} call implied volatility by strike ({expiration})')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig('hw06_iv_smile.png', dpi=160)
plt.show()

try:
    vix_data = yf.Ticker('^VIX').history(period='5d')
    vix = float(vix_data['Close'].iloc[-1])
    print(f'Latest available VIX: {vix:.2f} (index points, approximately percent)')
except Exception as error:
    print('VIX could not be retrieved:', error)

print(f'{ticker} latest close: ${spot:.2f}')
print(f'Option expiration: {expiration}')
print(f'Nearest listed call strike: ${strikes[near]:.2f}')
print(f'At-the-money implied volatility: {atm_vol:.2%}')
print(f'30-day annualized realized volatility: {realized_vol:.2%}')
print('Implied volatility higher than realized?', atm_vol > realized_vol)
