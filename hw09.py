"""Homework 9 starter — Algory QI Education, Fall 2026"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def recovery_needed(loss_fraction):
    """Gain required to get back to even after a loss.

    recovery_needed(0.50) -> 1.00     lose half, need to double
    recovery_needed(0.75) -> 3.00     lose three quarters, need +300%
    """
    return loss_fraction / (1 - loss_fraction)


def annualise_vol(daily_vol, days=252):
    """Scale a daily standard deviation to annual. Assumes days are independent."""
    return daily_vol * np.sqrt(days)


def var(returns, confidence=0.95):
    """Value at risk, as a return. A left-tail quantile, so it comes out negative."""
    return np.percentile(returns, (1 - confidence) * 100)


def cvar(returns, confidence=0.95):
    """Mean of everything at or beyond the VaR threshold.

    Must always be worse than var() on the same inputs. If it is not, you have
    averaged the wrong side of the distribution.
    """
    cutoff = var(returns, confidence)
    return np.mean(np.asarray(returns)[np.asarray(returns) <= cutoff])


def simulate_paths(start, drift, vol, days=252, paths=1000, seed=0):
    """Geometric Brownian motion. Return the FINAL value of each path."""
    np.random.seed(seed)
    final_values = np.zeros(paths)
    for i in range(paths):
        value = start
        for j in range(days):
            z = np.random.normal(0, 1)
            value = value * np.exp((drift - vol ** 2 / 2) / 252 + vol * np.sqrt(1 / 252) * z)
        final_values[i] = value
    return final_values


def main():
    data = yf.download('SPY', period='20y', auto_adjust=True, progress=False)
    if data.empty:
        print('Could not download SPY data.')
        return
    prices = data['Close']
    if prices.ndim > 1:
        prices = prices.iloc[:, 0]
    prices = prices.dropna()
    returns = prices.pct_change().dropna()

    # Q1 and Q2
    daily_vol = returns.std(ddof=1)
    scaled_vol = annualise_vol(daily_vol)
    year_prices = prices.resample('YE').last()
    year_returns = year_prices.pct_change().dropna()
    # The current calendar year is incomplete, so leave it out.
    if year_returns.index[-1].year == prices.index[-1].year:
        year_returns = year_returns.iloc[:-1]
    direct_vol = year_returns.std(ddof=1)
    ratio = direct_vol / scaled_vol
    print('Q1')
    print(f'Daily volatility: {daily_vol:.4%}')
    print(f'Scaled annual volatility: {scaled_vol:.4%}')
    print('Q2')
    print(f'Direct annual volatility: {direct_vol:.4%}')
    print(f'Scaled annual volatility: {scaled_vol:.4%}')
    print(f'Direct / scaled ratio: {ratio:.4f}')

    # Q3
    rolling_vol = returns.rolling(60).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    print('Q3')
    print(f'Highest rolling volatility: {rolling_vol.max():.4%}, {rolling_vol.idxmax().date()}')
    print(f'Lowest rolling volatility: {rolling_vol.min():.4%}, {rolling_vol.idxmin().date()}')
    # Find two distinct spikes instead of reporting adjacent days in the same spike.
    spike_dates = []
    for date in rolling_vol.sort_values(ascending=False).index:
        location = rolling_vol.index.get_loc(date)
        if all(abs(location - rolling_vol.index.get_loc(old)) >= 120 for old in spike_dates):
            spike_dates.append(date)
        if len(spike_dates) == 2:
            break
    for i, date in enumerate(spike_dates):
        print(f'Spike {i + 1}: {date.date()}, {rolling_vol.loc[date]:.4%}')
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_vol.index, rolling_vol.values)
    plt.title('SPY 60-Day Rolling Annualized Volatility')
    plt.xlabel('Date')
    plt.ylabel('Annualized Volatility')
    plt.tight_layout()
    plt.savefig('hw09_rolling_volatility.png', dpi=180)
    plt.close()

    # Q4 and Q5
    mean = returns.mean()
    std = returns.std(ddof=1)
    centered = returns - mean
    skewness = np.mean(centered ** 3) / np.mean(centered ** 2) ** 1.5
    kurtosis = np.mean(centered ** 4) / np.mean(centered ** 2) ** 2 - 3
    actual = int((abs(centered) > 4 * std).sum())
    # Normal two-sided probability beyond 4 sigma is about 0.0000633425.
    expected = len(returns) * 0.0000633424836662
    print('Q4')
    print(f'Skewness: {skewness:.4f}')
    print(f'Excess kurtosis: {kurtosis:.4f}')
    print(f'Four-sigma days: {actual}')
    print('Q5')
    print(f'Expected four-sigma days: {expected:.4f}')
    print(f'Actual four-sigma days: {actual}')
    print(f'Actual / expected: {actual / expected:.2f}')
    x = np.linspace(mean - 6 * std, mean + 6 * std, 500)
    normal = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.figure(figsize=(10, 5))
    plt.hist(returns, bins=100, density=True, alpha=0.6, label='SPY returns')
    plt.plot(x, normal, label='Fitted normal distribution')
    plt.title('SPY Daily Returns vs. Normal Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('hw09_return_histogram.png', dpi=180)
    plt.close()

    # Q6
    losses = [0.10, 0.25, 0.50, 0.75, 0.90]
    gains = [recovery_needed(loss) for loss in losses]
    print('Q6')
    for i in range(len(losses)):
        print(f'Loss: {losses[i]:.0%}, recovery gain: {gains[i]:.0%}')
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(losses) * 100, np.array(gains) * 100, marker='o')
    plt.title('Gain Needed to Recover from a Loss')
    plt.xlabel('Loss (%)')
    plt.ylabel('Required Gain (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hw09_recovery.png', dpi=180)
    plt.close()

    # Q7: use annualized log-return drift for geometric Brownian motion.
    log_returns = np.log(1 + returns)
    drift = log_returns.mean() * 252 + scaled_vol ** 2 / 2
    start = 100000
    simulated = simulate_paths(start, drift, scaled_vol, days=252, paths=10000, seed=0)
    print('Q7')
    print(f'Mean final value: ${simulated.mean():,.2f}')
    print(f'Median final value: ${np.median(simulated):,.2f}')
    print(f'5th percentile: ${np.percentile(simulated, 5):,.2f}')
    print(f'95th percentile: ${np.percentile(simulated, 95):,.2f}')

    # Q8
    sim_returns = simulated / start - 1
    sim_var = var(sim_returns)
    sim_cvar = cvar(sim_returns)
    print('Q8')
    print(f'GBM VaR: ${-sim_var * start:,.2f}, {-sim_var:.2%}')
    print(f'GBM CVaR: ${-sim_cvar * start:,.2f}, {-sim_cvar:.2%}')

    # Q9: sample historical daily returns with replacement.
    np.random.seed(0)
    bootstrap = np.zeros(10000)
    daily_values = returns.to_numpy()
    for i in range(10000):
        sampled = np.random.choice(daily_values, size=252, replace=True)
        bootstrap[i] = start * np.prod(1 + sampled)
    boot_returns = bootstrap / start - 1
    boot_var = var(boot_returns)
    boot_cvar = cvar(boot_returns)
    print('Q9')
    print(f'Bootstrap mean: ${bootstrap.mean():,.2f}')
    print(f'Bootstrap median: ${np.median(bootstrap):,.2f}')
    print(f'Bootstrap 5th percentile: ${np.percentile(bootstrap, 5):,.2f}')
    print(f'Bootstrap 95th percentile: ${np.percentile(bootstrap, 95):,.2f}')
    print(f'{"Method":<15} {"VaR ($)":>14} {"CVaR ($)":>14} {"VaR (%)":>12} {"CVaR (%)":>12}')
    print(f'{"GBM":<15} {-sim_var * start:>14,.2f} {-sim_cvar * start:>14,.2f} {-sim_var:>12.2%} {-sim_cvar:>12.2%}')
    print(f'{"Bootstrap":<15} {-boot_var * start:>14,.2f} {-boot_cvar * start:>14,.2f} {-boot_var:>12.2%} {-boot_cvar:>12.2%}')

    # Save the written answers using the values calculated above.
    bigger = 'bootstrap' if boot_var < sim_var else 'GBM' if sim_var < boot_var else 'neither method'
    direction = 'higher' if ratio > 1 else 'lower' if ratio < 1 else 'the same'
    with open('hw09_written_answers.txt', 'w') as file:
        file.write('Q2\n')
        file.write(f'The annual volatility measured directly was {direct_vol:.2%}, compared to {scaled_vol:.2%} using the square-root rule. The direct figure was {direction} and the ratio was {ratio:.3f}. This shows that scaling daily volatility does not necessarily give the same result as measuring annual returns directly. The square-root rule assumes daily returns are independent, which is not always true in actual markets.\n\n')
        file.write('Q3\n')
        file.write('The two largest distinct rolling-volatility spikes in my data were around ' + ' and '.join(str(date.date()) for date in spike_dates) + '. The chart shows that periods of high volatility tend to occur together rather than being spread evenly over time.\n\n')
        file.write('Q5\n')
        file.write(f'A normal distribution predicts about {expected:.2f} four-sigma days in this sample, but there were actually {actual}. That is about {actual / expected:.1f} times as many. This suggests that extreme returns happen much more often than a normal distribution would predict.\n\n')
        file.write('Q7\n')
        file.write(f'The mean ending portfolio value was ${simulated.mean():,.2f}, while the median was ${np.median(simulated):,.2f}. I think the median is more useful for understanding a typical outcome because a few unusually high ending values can pull the mean upward. They differ because the simulated ending values are not distributed symmetrically. The mean is still useful if you care about the average across many outcomes.\n\n')
        file.write('Q10\n')
        file.write(f'The {bigger} method produced the larger 95% VaR in this run. GBM assumes normally distributed log returns, while bootstrapping samples from the historical daily returns, including the extreme moves in the data. That difference can change the size of the estimated losses. VaR tells us the loss threshold at 95% confidence, but it does not tell us how severe losses are once we go past that threshold. CVaR addresses this by averaging the losses in the worst 5% of outcomes, so it gives a better idea of the risk in the tail.\n')
    print('Saved plots and written answers to the current folder.')


if __name__ == "__main__":
    main()
