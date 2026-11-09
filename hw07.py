"""Homework 7 starter — Algory QI Education, Fall 2026

The check that matters here is that your features never read the future. The
checker builds your features twice, changing only data after row 400, and
confirms the earlier rows did not move.
"""
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def get_close(prices):
    """Get a one-dimensional list of closing prices."""
    if hasattr(prices, "columns"):
        if "Close" in prices.columns:
            values = prices["Close"]
        else:
            values = prices["Adj Close"]
        return np.asarray(values, dtype=float).reshape(-1)
    return np.asarray(prices, dtype=float).reshape(-1)


def build_features(prices, lookbacks=(1, 5, 20, 60)):
    """One row per trading day: features first, label last.

    Every feature must use only prices up to and including that day. The label is
    the NEXT day's return, and it is the only forward-looking value allowed.

    Return a list of rows, each [f1, f2, ..., fn, label].
    """
    close = get_close(prices)
    rows = []
    start = max(max(lookbacks), 50, 20)

    for i in range(start, len(close) - 1):
        if not np.all(np.isfinite(close[i - start:i + 2])):
            continue
        if np.any(close[i - start:i + 2] <= 0):
            continue

        features = []
        for days in lookbacks:
            features.append(close[i] / close[i - days] - 1)

        daily_returns = []
        for j in range(i - 19, i + 1):
            daily_returns.append(close[j] / close[j - 1] - 1)
        volatility = np.std(daily_returns)
        features.append(volatility)

        moving_average = np.mean(close[i - 49:i + 1])
        features.append(close[i] / moving_average - 1)

        next_return = close[i + 1] / close[i] - 1
        rows.append(features + [next_return])

    return rows


def split_by_time(rows, train=0.6, val=0.2):
    """Split in time order into (training, validation, test). Never shuffle.

    Shuffling lets 2024 train the model that predicts 2016. It produces
    spectacular, meaningless results and it is the most common mistake in the field.
    """
    train_end = int(len(rows) * train)
    val_end = train_end + int(len(rows) * val)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def up_day_baseline(prices):
    """Share of days with a positive return: the number any classifier must beat.

    It is nearer 54% than 50%, which is why raw accuracy is a weak measure here.
    """
    close = get_close(prices)
    changes = np.diff(close)
    return float(np.mean(changes > 0)) if len(changes) else 0.0


def backtest(returns, predictions):
    """Walk forward using model predictions, starting with $1 in each portfolio."""
    returns = np.asarray(returns, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(returns) == 0 or len(returns) != len(predictions):
        raise ValueError("Returns and predictions must have equal nonzero lengths")

    portfolio = 1.0
    buy_hold = 1.0
    portfolio_values = [1.0]
    buy_hold_values = [1.0]
    days_invested = 0

    for i in range(len(returns)):
        if predictions[i] > 0:
            portfolio = portfolio * (1 + returns[i])
            days_invested += 1

        buy_hold = buy_hold * (1 + returns[i])
        portfolio_values.append(portfolio)
        buy_hold_values.append(buy_hold)

    portfolio_values = np.array(portfolio_values)
    buy_hold_values = np.array(buy_hold_values)

    strategy_annual = portfolio ** (252 / len(returns)) - 1
    buy_hold_annual = buy_hold ** (252 / len(returns)) - 1

    strategy_peak = np.maximum.accumulate(portfolio_values)
    buy_hold_peak = np.maximum.accumulate(buy_hold_values)

    strategy_drawdown = np.min(portfolio_values / strategy_peak - 1)
    buy_hold_drawdown = np.min(buy_hold_values / buy_hold_peak - 1)
    fraction_invested = days_invested / len(returns)

    return (strategy_annual, buy_hold_annual,
            strategy_drawdown, buy_hold_drawdown,
            fraction_invested)


def main():
    # TODO Q1: build the table, print its shape, columns and first three rows
    prices = yf.download("SPY", period="15y", auto_adjust=True, progress=False)
    if prices.empty:
        print("Could not download SPY prices.")
        return

    close = get_close(prices)
    rows = build_features(close)
    columns = ["return_1", "return_5", "return_20", "return_60",
               "volatility_20", "price_vs_ma50", "next_day_return"]
    print("Q1: Building the dataset")
    print("Table shape:", np.shape(rows))
    print("Columns:", columns)
    for row in rows[:3]:
        print(dict(zip(columns, np.round(row, 6))))
    print("Up-day baseline:", round(up_day_baseline(close), 4))

    # Check one row by hand: the first 1-day return
    first_day = 60
    print("First row 1-day return check:", close[first_day] / close[first_day - 1] - 1)

    # TODO Q2: split, print the date range and row count of each set
    train_rows, val_rows, test_rows = split_by_time(rows)
    dates = prices.index[60:-1]
    train_end = len(train_rows)
    val_end = train_end + len(val_rows)
    print("\nQ2: Splitting the data")
    for name, subset, date_part in [
        ("Training", train_rows, dates[:train_end]),
        ("Validation", val_rows, dates[train_end:val_end]),
        ("Test", test_rows, dates[val_end:])
    ]:
        print(name, "rows:", len(subset), "dates:", date_part[0].date(), "to", date_part[-1].date())

    train_data = np.array(train_rows)
    val_data = np.array(val_rows)
    test_data = np.array(test_rows)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_val, y_val = val_data[:, :-1], val_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # TODO Q3: four models, training and validation R-squared for each
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0)
    }
    print("\nQ3: Model results")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(name, "train R2:", round(model.score(X_train, y_train), 5),
              "validation R2:", round(model.score(X_val, y_val), 5))

    # TODO Q4: tune one hyperparameter each, plot the tuning curve
    choices = {
        "Ridge": [0.01, 0.1, 1, 10, 100, 1000],
        "Random Forest": [2, 4, 6, 8, 12, 20],
        "Gradient Boosting": [10, 25, 50, 75, 100, 150]
    }
    tuned = {}
    print("\nQ4: Tuning")
    for name, values in choices.items():
        scores = []
        for value in values:
            if name == "Ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=value))
            elif name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, max_depth=value,
                                              random_state=0, n_jobs=-1)
            else:
                model = GradientBoostingRegressor(n_estimators=value, random_state=0)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))

        best_index = int(np.argmax(scores))
        best_value = values[best_index]
        if name == "Ridge":
            best_model = make_pipeline(StandardScaler(), Ridge(alpha=best_value))
        elif name == "Random Forest":
            best_model = RandomForestRegressor(n_estimators=100, max_depth=best_value,
                                                random_state=0, n_jobs=-1)
        else:
            best_model = GradientBoostingRegressor(n_estimators=best_value, random_state=0)
        best_model.fit(X_train, y_train)
        tuned[name] = best_model
        print(name, "best value:", best_value, "validation R2:", round(scores[best_index], 5))

        plt.figure()
        plt.plot(values, scores, marker="o")
        plt.xlabel({"Ridge": "Alpha", "Random Forest": "Maximum depth",
                    "Gradient Boosting": "Number of estimators"}[name])
        plt.ylabel("Validation R-squared")
        plt.title(name + " Hyperparameter Tuning")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(name.lower().replace(" ", "_") + "_tuning.png")

    # TODO Q5: best model, once, on the test set
    all_models = {"Linear Regression": models["Linear Regression"]}
    all_models.update(tuned)
    best_name = max(all_models, key=lambda name: all_models[name].score(X_val, y_val))
    best_model = all_models[best_name]
    predictions = best_model.predict(X_test)
    test_r2 = best_model.score(X_test, y_test)
    correlation = np.corrcoef(predictions, y_test)[0, 1]
    print("\nQ5: Test results")
    print("Best model:", best_name)
    print("Test R2:", round(test_r2, 5))
    print("Prediction correlation:", round(correlation, 5))

    # TODO Q6: convert to a trading rule and backtest it
    strategy_annual, buy_hold_annual, strategy_drawdown, buy_hold_drawdown, fraction_invested = backtest(y_test, predictions)
    print("\nQ6: Backtest (no fees or transaction costs)")
    print("Strategy annualized return:", round(strategy_annual * 100, 2), "%",
          "max drawdown:", round(strategy_drawdown * 100, 2), "%",
          "days invested:", round(fraction_invested * 100, 2), "%")
    print("Buy and hold annualized return:", round(buy_hold_annual * 100, 2), "%",
          "max drawdown:", round(buy_hold_drawdown * 100, 2), "%",
          "days invested: 100.0 %")

    # TODO Q7: rank the features the model leaned on
    print("\nQ7: Feature ranking")
    feature_names = columns[:-1]
    if best_name in ("Linear Regression", "Ridge"):
        weights = best_model[-1].coef_
    else:
        weights = best_model.feature_importances_
    order = np.argsort(np.abs(weights))[::-1]
    for index in order:
        print(feature_names[index], round(weights[index], 8))

    plt.show()


if __name__ == "__main__":
    main()
