"""Risk analytics on weekly data: volatility, category portfolio and Value at Risk."""

from statistics import NormalDist

import numpy as np
import pandas as pd


def _complete_week_index(start, end):
    """Week labels (Sundays) of the weeks that lie fully inside [start, end]."""
    sundays = pd.date_range(start, end + pd.Timedelta(days=6), freq="W")
    return sundays[(sundays - pd.Timedelta(days=6) >= start) & (sundays <= end)]


def weekly_series(sales, start, end, value="revenue"):
    """Weekly totals of one column, only complete weeks, weeks without sales are 0."""
    weeks = _complete_week_index(start, end)
    weekly = sales.set_index("sale_date")[value].resample("W").sum()
    return weekly.reindex(weeks, fill_value=0.0)


def weekly_by_category(sales, start, end, value="profit"):
    """Weekly totals per product category, only complete weeks."""
    weeks = _complete_week_index(start, end)
    pivot = sales.pivot_table(index="sale_date", columns="category", values=value, aggfunc="sum")
    return pivot.resample("W").sum().reindex(weeks, fill_value=0.0)


def weekly_returns(weekly):
    """Week over week percentage change, undefined weeks (after a zero week) are dropped."""
    return weekly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def rolling_volatility(returns, window):
    return returns.rolling(window).std().dropna()


def risk_return(cat_weekly):
    """Mean weekly profit as return and its standard deviation as risk, per category."""
    table = pd.DataFrame({"mean_profit": cat_weekly.mean(), "std_profit": cat_weekly.std()})
    table["return_per_risk"] = table["mean_profit"] / table["std_profit"].replace(0, np.nan)
    return table


def diversification(cat_weekly):
    """Risk of the combined portfolio compared with the sum of the category risks."""
    portfolio_std = cat_weekly.sum(axis=1).std()
    sum_std = cat_weekly.std().sum()
    return {
        "portfolio_std": portfolio_std,
        "sum_std": sum_std,
        "reduction": 1 - portfolio_std / sum_std if sum_std else 0.0,
    }


def value_at_risk(returns, confidence, avg_revenue):
    """Historical and parametric (normal) VaR of weekly revenue returns."""
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(1 - confidence)
    hist = returns.quantile(1 - confidence)
    param = returns.mean() + z * returns.std()
    return pd.DataFrame(
        {
            "VaR (return)": [hist, param],
            "VaR (revenue)": [hist * avg_revenue, param * avg_revenue],
        },
        index=["Historical", "Parametric"],
    )
