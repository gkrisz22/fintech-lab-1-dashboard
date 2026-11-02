"""Business performance KPIs and revenue breakdowns."""

import pandas as pd


def kpis(sales, start, end, data_start):
    """Headline numbers for [start, end] and growth against the previous period of the same length.

    `sales` must be filtered by category and country but not by date, so the
    previous period can be taken from it.
    """
    current = sales[sales["sale_date"].between(start, end)]
    revenue = current["revenue"].sum()
    transactions = current["invoice_id"].nunique()

    length = end - start + pd.Timedelta(days=1)
    prev_start, prev_end = start - length, start - pd.Timedelta(days=1)
    growth = None
    if prev_start >= data_start:
        prev_revenue = sales.loc[sales["sale_date"].between(prev_start, prev_end), "revenue"].sum()
        if prev_revenue > 0:
            growth = revenue / prev_revenue - 1

    return {
        "revenue": revenue,
        "growth": growth,
        "previous_period": (prev_start, prev_end),
        "customers": current["customer_id"].nunique(),
        "transactions": transactions,
        "avg_transaction": revenue / transactions if transactions else 0.0,
    }


def monthly_revenue(sales):
    """Monthly revenue with a 3 month moving average."""
    monthly = sales.set_index("sale_date")["revenue"].resample("ME").sum().to_frame()
    monthly["ma3"] = monthly["revenue"].rolling(3).mean()
    return monthly


def top_products(sales, n):
    table = (
        sales.groupby(["product_name", "category"], as_index=False)
        .agg(revenue=("revenue", "sum"), units=("quantity", "sum"), profit=("profit", "sum"))
        .sort_values("revenue", ascending=False)
    )
    return table.head(n)


def top_customers(sales, n):
    table = (
        sales.groupby(["customer_name", "country"], as_index=False)
        .agg(revenue=("revenue", "sum"), transactions=("invoice_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )
    return table.head(n)


def category_revenue(sales):
    table = (
        sales.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), profit=("profit", "sum"))
        .sort_values("revenue", ascending=False)
    )
    table["share"] = table["revenue"] / table["revenue"].sum()
    return table
