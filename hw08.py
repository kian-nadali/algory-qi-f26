"""Homework 8 starter — Algory QI Education, Fall 2026"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis


def n_day_returns(prices, n):

    prices = np.asarray(prices, dtype=float)
    if n < 1 or len(prices) <= n:
        return np.array([])
    returns = []
    for i in range(len(prices) - n):
        returns.append(prices[i + n] / prices[i] - 1)
    return np.array(returns)


def price_from_history(prices, spot, strike, days, kind):
 
    returns = n_day_returns(prices, days)
    if len(returns) == 0:
        return np.nan
    final_prices = spot * (1 + returns)
    if kind == 'call':
        payoffs = np.maximum(final_prices - strike, 0)
    elif kind == 'put':
        payoffs = np.maximum(strike - final_prices, 0)
    else:
        raise ValueError('kind must be call or put')
    return float(np.mean(payoffs))


def black_scholes(spot, strike, time, rate, volatility, kind):
    # Same function from Homework 6
    if time <= 0:
        if kind == 'call':
            return max(spot - strike, 0)
        return max(strike - spot, 0)
    if volatility <= 0:
        final = spot * np.exp(rate * time)
        if kind == 'call':
            return np.exp(-rate * time) * max(final - strike, 0)
        return np.exp(-rate * time) * max(strike - final, 0)
    d1 = (np.log(spot / strike) + (rate + volatility ** 2 / 2) * time) / (volatility * np.sqrt(time))
    d2 = d1 - volatility * np.sqrt(time)
    if kind == 'call':
        return spot * norm.cdf(d1) - strike * np.exp(-rate * time) * norm.cdf(d2)
    return strike * np.exp(-rate * time) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def main():
    # Q1
    history = yf.Ticker('SPY').history(period='20y', auto_adjust=True)
    prices = history['Close'].dropna().to_numpy()
    if len(prices) < 100:
        raise RuntimeError('Not enough SPY history returned by Yahoo Finance.')
    days = 21
    returns = n_day_returns(prices, days)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    print('Q1')
    print('Number of windows:', len(returns))
    print(f'Mean: {mean:.4%}')
    print(f'Standard deviation: {std:.4%}')
    print(f'1st percentile: {np.percentile(returns, 1):.4%}')
    print(f'99th percentile: {np.percentile(returns, 99):.4%}')

    # Q2
    print('\nQ2')
    print(f'Skewness: {skew(returns):.4f}')
    print(f'Excess kurtosis: {kurtosis(returns):.4f}')
    x = np.linspace(min(returns), max(returns), 500)
    plt.figure(figsize=(9, 5))
    plt.hist(returns, bins=60, density=True, alpha=0.6, label='Historical returns')
    plt.plot(x, norm.pdf(x, mean, std), label='Normal distribution')
    plt.xlabel('21-day return')
    plt.ylabel('Density')
    plt.title('SPY historical 21-day returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hw08_returns.png', dpi=160)
    plt.show()

    # Q3
    spot = 760.88
    strike = spot * 1.03
    call = price_from_history(prices, spot, strike, days, 'call')
    put = price_from_history(prices, spot, strike, days, 'put')
    print('\nQ3')
    print(f'Spot: ${spot:.2f}; strike: ${strike:.2f}')
    print(f'Historical call: ${call:.4f}')
    print(f'Historical put: ${put:.4f}')
    print('Historical prices do not include risk-free discounting.')

    # Q4
    final_prices = spot * (1 + returns)
    print('\nQ4')
    print(f'Call in-the-money fraction: {np.mean(final_prices > strike):.4%}')
    print(f'Put in-the-money fraction: {np.mean(final_prices < strike):.4%}')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, kind in zip(axes, ['call', 'put']):
        ax.hist(final_prices, bins=60, alpha=0.65)
        ax.axvline(strike, color='black', linestyle='--', label='Strike')
        if kind == 'call':
            ax.axvspan(strike, max(final_prices), alpha=0.2, color='green', label='Pays off')
        else:
            ax.axvspan(min(final_prices), strike, alpha=0.2, color='orange', label='Pays off')
        ax.set_title(kind.capitalize() + ' final prices')
        ax.set_xlabel('Final SPY price ($)')
        ax.set_ylabel('Windows')
        ax.legend()
    plt.tight_layout()
    plt.savefig('hw08_final_prices.png', dpi=160)
    plt.show()

    # Q5
    daily = n_day_returns(prices, 1)
    realized_vol = np.std(daily[-60:], ddof=1) * np.sqrt(252)
    rate = 0.04  # Same assumed annual rate used in Homework 6
    time = days / 252
    bs_call = black_scholes(spot, strike, time, rate, realized_vol, 'call')
    bs_put = black_scholes(spot, strike, time, rate, realized_vol, 'put')
    print('\nQ5')
    print(f'60-day annualized realized volatility: {realized_vol:.4%}')
    for kind, historical, bs in [('Call', call, bs_call), ('Put', put, bs_put)]:
        difference = (historical - bs) / bs * 100 if bs != 0 else np.nan
        print(f'{kind}: historical ${historical:.4f}, Black-Scholes ${bs:.4f}, difference {difference:.2f}%')

    # Q6
    strikes = np.linspace(spot * 0.85, spot * 1.15, 101)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, kind in enumerate(['call', 'put']):
        historical = np.array([price_from_history(prices, spot, k, days, kind) for k in strikes])
        bs = np.array([black_scholes(spot, k, time, rate, realized_vol, kind) for k in strikes])
        difference = historical - bs
        axes[0, col].plot(strikes, historical, label='Historical')
        axes[0, col].plot(strikes, bs, label='Black-Scholes')
        axes[0, col].set_title(kind.capitalize() + ' prices')
        axes[0, col].set_ylabel('Option price ($)')
        axes[0, col].legend()
        axes[1, col].plot(strikes, difference)
        axes[1, col].axhline(0, color='black', linewidth=0.8)
        axes[1, col].set_title(kind.capitalize() + ': historical minus Black-Scholes')
        axes[1, col].set_xlabel('Strike ($)')
        axes[1, col].set_ylabel('Difference ($)')
        for row in range(2):
            axes[row, col].grid(alpha=0.25)
        idx = np.argmax(np.abs(difference))
        print(f'Q6 {kind}: largest absolute difference at strike ${strikes[idx]:.2f}: ${difference[idx]:+.4f}')
    plt.tight_layout()
    plt.savefig('hw08_comparison.png', dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
