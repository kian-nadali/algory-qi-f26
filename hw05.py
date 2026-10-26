import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def intrinsic(kind, strike, spot):
    if kind == "call":
        return np.maximum(spot - strike, 0)

    elif kind == "put":
        return np.maximum(strike - spot, 0)


def extrinsic(kind, strike, spot, premium):
    return premium - intrinsic(kind, strike, spot)


def payoff(kind, strike, premium, spot):
    if kind == "long_call":
        return intrinsic("call", strike, spot) - premium

    elif kind == "long_put":
        return intrinsic("put", strike, spot) - premium

    elif kind == "short_call":
        return premium - intrinsic("call", strike, spot)

    elif kind == "short_put":
        return premium - intrinsic("put", strike, spot)


def main():
    # ---------------------------------------------------------------- Q1
    kinds = ["long_call", "long_put", "short_call", "short_put"]

    for kind in kinds:
        for spot in [80, 100, 120]:
            print("Q1", kind, "spot:", spot,
                  "profit:", payoff(kind, 100, 6, spot))

    # ---------------------------------------------------------------- Q2
    spots = np.linspace(60, 140, 100)

    for i in range(len(kinds)):
        plt.subplot(2, 2, i + 1)
        plt.plot(spots, payoff(kinds[i], 100, 6, spots))
        plt.axvline(x=100, linestyle="--", color="gray")
        plt.axhline(y=0, color="black")
        plt.xlabel("Spot Price at Expiry")
        plt.ylabel("Profit per Share")
        plt.title(kinds[i])

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------- Q3
    ticker = "SPY"
    t = yf.Ticker(ticker)

    print("Q3 available expiry dates:", t.options)

    expiry = "2026-11-27"
    chain = t.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts

    history = t.history(period="5d")
    spot = history["Close"].iloc[-1]

    print("Q3 ticker:", ticker)
    print("Q3 expiry:", expiry)
    print("Q3 spot price (latest close):", spot)
    print("Q3 number of calls:", len(calls))
    print("Q3 number of puts:", len(puts))

    # ---------------------------------------------------------------- Q4
    calls = calls[calls["bid"] > 0].copy()

    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    calls["intrinsic"] = intrinsic("call", calls["strike"], spot)
    calls["extrinsic"] = extrinsic(
        "call", calls["strike"], spot, calls["mid"]
    )

    calls["distance"] = abs(calls["strike"] - spot)
    nearest = calls.nsmallest(10, "distance")
    nearest = nearest.sort_values("strike")

    print("Q4 ten strikes nearest to spot:")
    print(nearest[["strike", "mid", "intrinsic", "extrinsic"]].round(2))

    # ---------------------------------------------------------------- Q5
    calls = calls.sort_values("strike")

    plt.plot(calls["strike"], calls["extrinsic"])
    plt.axvline(x=spot, linestyle="--", color="black")
    plt.xlabel("Strike Price")
    plt.ylabel("Extrinsic Value per Share")
    plt.title("SPY Call Extrinsic Value: " + expiry)
    plt.show()

    largest = calls["extrinsic"].idxmax()

    print("Q5 strike with largest extrinsic value:",
          calls.loc[largest, "strike"])
    print("Q5 spot price:", spot)

    # ---------------------------------------------------------------- Q6
    long_expiry = "2027-06-17"
    long_chain = t.option_chain(long_expiry)
    long_calls = long_chain.calls

    long_calls = long_calls[long_calls["bid"] > 0].copy()

    long_calls["mid"] = (long_calls["bid"] + long_calls["ask"]) / 2
    long_calls["intrinsic"] = intrinsic(
        "call", long_calls["strike"], spot
    )
    long_calls["extrinsic"] = extrinsic(
        "call", long_calls["strike"], spot, long_calls["mid"]
    )

    long_calls["distance"] = abs(long_calls["strike"] - spot)
    long_nearest = long_calls.nsmallest(10, "distance")
    long_nearest = long_nearest.sort_values("strike")

    print("Q6 long expiry:", long_expiry)
    print("Q6 ten strikes nearest to spot:")
    print(long_nearest[
        ["strike", "mid", "intrinsic", "extrinsic"]
    ].round(2))

    long_calls = long_calls.sort_values("strike")

    plt.plot(calls["strike"], calls["extrinsic"], label=expiry)
    plt.plot(long_calls["strike"], long_calls["extrinsic"],
             label=long_expiry)
    plt.axvline(x=spot, linestyle="--", color="black", label="Spot")
    plt.xlabel("Strike Price")
    plt.ylabel("Extrinsic Value per Share")
    plt.title("SPY Call Extrinsic Value at Two Expiries")
    plt.legend()
    plt.show()

    long_largest = long_calls["extrinsic"].idxmax()

    print("Q6 short expiry largest extrinsic value:",
          calls["extrinsic"].max())
    print("Q6 long expiry largest extrinsic value:",
          long_calls["extrinsic"].max())
    print("Q6 long expiry strike with largest extrinsic value:",
          long_calls.loc[long_largest, "strike"])
    print("Q6 spot price:", spot)


if __name__ == "__main__":
    main()