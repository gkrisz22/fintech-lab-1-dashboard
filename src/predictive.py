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
    """Refit on all weeks and forecast `horizon` weeks ahead, feeding predictions back as lags."""
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
