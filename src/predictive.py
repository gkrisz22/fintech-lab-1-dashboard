"""Weekly revenue forecasting with lag features and scikit-learn regressors."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

MIN_WEEKS = 30
MIN_TRAIN_ROWS = 20
LAGS = [1, 2, 3, 4]
FEATURES = [f"lag_{lag}" for lag in LAGS] + ["rolling_mean_4", "month"]
MODEL_NAMES = ["Linear Regression", "Random Forest"]


def make_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42)
    raise ValueError(f"unknown model: {name}")


def make_features(weekly):
    """Previous weeks' revenue and the month as features for each week."""
    df = pd.DataFrame({"revenue": weekly})
    for lag in LAGS:
        df[f"lag_{lag}"] = weekly.shift(lag)
    df["rolling_mean_4"] = weekly.shift(1).rolling(4).mean()
    df["month"] = weekly.index.month
    return df.dropna()


def check_series(weekly, test_weeks):
    """Raise ValueError when there is not enough history for the chosen split."""
    if len(weekly) < MIN_WEEKS:
        raise ValueError(
            f"Forecasting needs at least {MIN_WEEKS} complete weeks, the current filter leaves {len(weekly)}. "
            "Widen the date range."
        )
    train_rows = len(weekly) - len(LAGS) - test_weeks
    if train_rows < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {train_rows} weeks are left for training. Lower the number of test weeks or widen the date range."
        )


def _metrics(actual, predicted):
    error = actual - predicted
    nonzero = actual != 0
    return {
        "MAE": error.abs().mean(),
        "RMSE": float(np.sqrt((error**2).mean())),
        "MAPE": (error[nonzero].abs() / actual[nonzero]).mean(),
        "R2": r2_score(actual, predicted),
    }


def evaluate(weekly, model_name, test_weeks):
    """Train on all but the last `test_weeks` weeks and predict those weeks one step ahead.

    Returns the predictions, a metrics table (model and naive baseline) and the
    standard deviation of the test errors.
    """
    check_series(weekly, test_weeks)
    data = make_features(weekly)
    train, test = data.iloc[:-test_weeks], data.iloc[-test_weeks:]

    model = make_model(model_name).fit(train[FEATURES], train["revenue"])
    result = pd.DataFrame(
        {
            "actual": test["revenue"],
            "predicted": model.predict(test[FEATURES]),
            "naive": test["lag_1"],
        },
        index=test.index,
    )
    metrics = pd.DataFrame(
        {
            model_name: _metrics(result["actual"], result["predicted"]),
            "Naive (last week)": _metrics(result["actual"], result["naive"]),
        }
    ).T
    error_std = (result["actual"] - result["predicted"]).std()
    return result, metrics, error_std


def forecast(weekly, model_name, horizon, error_std):
    """Refit on all weeks and forecast `horizon` weeks ahead, feeding predictions back as lags.

    `error_std` is one value for every week or an array with one value per week ahead.
    """
    if not 1 <= horizon <= 52:
        raise ValueError("horizon must be between 1 and 52 weeks")
    data = make_features(weekly)
    model = make_model(model_name).fit(data[FEATURES], data["revenue"])

    history = list(weekly.values)
    dates = pd.date_range(weekly.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    predictions = []
    for date in dates:
        row = {f"lag_{lag}": history[-lag] for lag in LAGS}
        row["rolling_mean_4"] = np.mean(history[-4:])
        row["month"] = date.month
        value = max(float(model.predict(pd.DataFrame([row])[FEATURES])[0]), 0.0)
        predictions.append(value)
        history.append(value)

    out = pd.DataFrame({"forecast": predictions}, index=dates)
    out["lower"] = (out["forecast"] - 1.96 * error_std).clip(lower=0)
    out["upper"] = out["forecast"] + 1.96 * error_std
    return out


def backtest_min_weeks(origins, horizon):
    """Weeks of history needed to backtest `origins` cut-off weeks with a `horizon` week forecast."""
    return MIN_TRAIN_ROWS + len(LAGS) + origins + horizon - 1


def backtest(weekly, model_name, origins, horizon):
    """Walk-forward test of the live forecast.

    At each of the last `origins` cut-off weeks the model is refit on the weeks before it and
    forecasts `horizon` weeks recursively, like `forecast`. Returns one row per forecast week with
    the actual value, the forecast and two baselines: the last known week and the same week a year earlier.
    """
    if len(weekly) < backtest_min_weeks(origins, horizon):
        raise ValueError(f"backtest needs at least {backtest_min_weeks(origins, horizon)} weeks")
    rows = []
    last = len(weekly) - horizon
    for cut in range(last - origins + 1, last + 1):
        train, actual = weekly.iloc[:cut], weekly.iloc[cut : cut + horizon]
        predicted = forecast(train, model_name, horizon, 0.0)["forecast"]
        rows.append(
            pd.DataFrame(
                {
                    "origin": train.index[-1],
                    "h": np.arange(1, horizon + 1),
                    "actual": actual.to_numpy(),
                    "predicted": predicted.to_numpy(),
                    "naive": train.iloc[-1],
                    "seasonal": weekly.reindex(actual.index - pd.Timedelta(weeks=52)).to_numpy(),
                },
                index=actual.index,
            )
        )
    return pd.concat(rows)


def horizon_rmse(bt):
    """Root mean squared forecast error for each week ahead, used as the width of the forecast band."""
    error = bt["actual"] - bt["predicted"]
    return np.sqrt((error**2).groupby(bt["h"]).mean())


def error_by_horizon(bt):
    """MAE of the model and the baselines for each week ahead.

    The seasonal baseline is left out when the data does not reach a year back for every forecast week.
    """
    columns = ["predicted", "naive"] + (["seasonal"] if bt["seasonal"].notna().all() else [])
    return bt[columns].sub(bt["actual"], axis=0).abs().groupby(bt["h"]).mean()


def band_coverage(bt, z=1.96):
    """Share of actual weeks inside the forecast band.

    The band for each cut-off week is built from the errors of the other cut-off weeks,
    so the check is not done on the same errors that set the band width.
    """
    inside = []
    for origin, group in bt.groupby("origin"):
        width = z * group["h"].map(horizon_rmse(bt[bt["origin"] != origin]))
        inside.append((group["actual"] - group["predicted"]).abs() <= width)
    return pd.concat(inside).mean()


def backtest_summary(bt):
    """Overall accuracy of the backtest forecasts."""
    error = (bt["actual"] - bt["predicted"]).abs()
    return {
        "accuracy": 1 - error.sum() / bt["actual"].sum(),
        "skill": 1 - error.mean() / (bt["actual"] - bt["naive"]).abs().mean(),
        "coverage": band_coverage(bt),
        "mae_next_week": error[bt["h"] == 1].mean(),
    }
