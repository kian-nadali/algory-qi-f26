import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- Q1
def simulate_dice(trials, seed=0):  
    np.random.seed(seed)

    four_sided_ones = 0
    total_ones = 0

    for i in range(trials):
        die = np.random.choice([4, 6])
        roll = np.random.randint(1, die + 1)

        if roll == 1:
            total_ones += 1

            if die == 4:
                four_sided_ones += 1

    probability = four_sided_ones / total_ones
    return probability


# ---------------------------------------------------------------- Q2
def simulate_coins(trials, seed=0):
    np.random.seed(seed)

    total_payout = 0

    for i in range(trials):
        heads = 0

        for j in range(3):
            flip = np.random.choice(["H", "T"])

            if flip == "H":
                heads += 1

        tails = 3 - heads
        payout = heads * tails
        total_payout += payout

    average_payout = total_payout / trials

    return average_payout


# ---------------------------------------------------------------- Q5/Q6
def p_down(returns):
    down_days = 0

    for r in returns:
        if r < 0:
            down_days += 1

    probability = down_days / len(returns)

    return probability, len(returns)


def p_down_given_down(returns):
    today_down = 0
    tomorrow_down = 0

    for i in range(len(returns) - 1):
        if returns.iloc[i] < 0:
            today_down += 1

            if returns.iloc[i + 1] < 0:
                tomorrow_down += 1

    probability = tomorrow_down / today_down

    return probability, today_down


def p_down_given_big_drop(returns, threshold=-0.02):
    big_drop_days = 0
    tomorrow_down = 0

    for i in range(len(returns) - 1):
        if returns.iloc[i] < threshold:
            big_drop_days += 1

            if returns.iloc[i + 1] < 0:
                tomorrow_down += 1

    probability = tomorrow_down / big_drop_days

    return probability, big_drop_days


# ---------------------------------------------------------------- Q7
def expected_present_value(cash_flows, rate, survival_prob):
    total = 0

    for i in range(len(cash_flows)):
        year = i + 1
        survival = survival_prob ** i
        discounted_cash_flow = cash_flows[i] / ((1 + rate) ** year)

        total += survival * discounted_cash_flow

    return total


def main():
    print("Q1  P(4-sided | rolled a 1) =", simulate_dice(100_000))
    print("Q2  expected three-coin payout =", simulate_coins(100_000))

    trial_counts = [100, 1000, 10000, 100000]
    estimates = []

    for trials in trial_counts:
        estimate = simulate_coins(trials)
        estimates.append(estimate)
        print("Q3", trials, "trials:", estimate)

    plt.plot(trial_counts, estimates, marker="o")
    plt.axhline(y=1.5, linestyle="--")
    plt.xlabel("Number of Trials")
    plt.ylabel("Estimated Expected Payout")
    plt.title("Three-Coin Simulation")
    plt.xscale("log")
    plt.show()

    spy = yf.Ticker("SPY").history(period="10y")
    spy_close = spy["Close"].dropna()
    returns = spy_close.pct_change().dropna()

    print("Q4 trading days:", len(returns))
    print("Q4 mean daily return:", returns.mean())

    prob_down, count_down = p_down(returns)
    prob_given_down, count_given_down = p_down_given_down(returns)
    prob_big_drop, count_big_drop = p_down_given_big_drop(returns)

    print("Q5 P(tomorrow is down):", prob_down, "count:", count_down)
    print("Q5 P(tomorrow is down | today was down):", prob_given_down, "count:", count_given_down)
    print("Q6 P(tomorrow is down | today was down more than 2%):", prob_big_drop, "count:", count_big_drop)
    print("Q7  EPV =", expected_present_value([10, 10, 10], 0.10, 0.5))


if __name__ == "__main__":
    main()
