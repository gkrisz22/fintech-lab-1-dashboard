"""Load the ERP data from the SQLite database and apply the dashboard filters."""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financial_analytics.db"
TABLES = ["customers", "products", "sales", "invoices", "payments"]


def load_tables(db_path=DB_PATH):
    """Read all tables and return them prepared for analysis."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"database file not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        tables = {name: pd.read_sql(f"SELECT * FROM {name}", con) for name in TABLES}
    finally:
        con.close()
    return prepare(tables)


def prepare(tables):
    """Convert dates and join sales lines with products, invoices and customers."""
    customers = tables["customers"].copy()
    products = tables["products"].copy()
    sales = tables["sales"].copy()
    invoices = tables["invoices"].copy()
    payments = tables["payments"].copy()

    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"])
    invoices["due_date"] = pd.to_datetime(invoices["due_date"])
    payments["payment_date"] = pd.to_datetime(payments["payment_date"])
    customers["customer_name"] = customers["first_name"] + " " + customers["last_name"]

    customer_cols = customers[["customer_id", "customer_name", "country"]]
    invoices = invoices.merge(customer_cols, on="customer_id", how="left")

    sales = (
        sales.merge(products[["product_id", "product_name", "category", "unit_cost"]], on="product_id")
        .merge(invoices[["invoice_id", "customer_id", "customer_name", "country"]], on="invoice_id")
    )
    sales["profit"] = sales["revenue"] - sales["quantity"] * sales["unit_cost"]

    return {
        "customers": customers,
        "products": products,
        "sales": sales,
        "invoices": invoices,
        "payments": payments,
    }


def parse_date_range(value):
    """Turn the date picker value into a (start, end) pair or raise ValueError."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Select both a start date and an end date.")
    start, end = pd.Timestamp(value[0]), pd.Timestamp(value[1])
    if start > end:
        raise ValueError("The start date must not be after the end date.")
    return start, end


def check_selection(selected, label):
    """Raise ValueError when a multiselect filter is empty."""
    if not selected:
        raise ValueError(f"Select at least one {label}.")


def filter_dimensions(sales, categories, countries):
    """Keep sales lines in the selected categories and customer countries."""
    mask = sales["category"].isin(categories) & sales["country"].isin(countries)
    return sales[mask]


def filter_dates(df, column, start, end):
    """Keep rows whose date column falls inside [start, end]."""
    return df[df[column].between(start, end)]
