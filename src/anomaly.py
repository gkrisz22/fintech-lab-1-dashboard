"""Invoice level anomaly detection with an Isolation Forest."""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "invoice_value": "invoice value",
    "lines": "number of lines",
    "units": "units",
    "avg_discount": "average discount",
    "paid_share": "paid share",
    "days_late": "days late",
    "payments": "number of payments",
}
REASON_Z = 2.0


def invoice_features(invoices, sales, payments):
    """One row per invoice with order size, discount and payment behaviour.

    Days late is measured from the due date to the last payment for settled
    invoices. Open invoices count as late until the last payment date in the data.
    """
    as_of = payments["payment_date"].max()
    lines = sales.groupby("invoice_id").agg(
        lines=("sale_id", "size"), units=("quantity", "sum"), avg_discount=("discount_rate", "mean")
    )
    paid = payments.groupby("invoice_id").agg(
        paid=("amount_paid", "sum"), payments=("payment_id", "size"), last_payment=("payment_date", "max")
    )

    df = invoices.set_index("invoice_id").join(lines).join(paid)
    df[["paid", "payments"]] = df[["paid", "payments"]].fillna(0)
    df["paid_share"] = df["paid"] / df["invoice_value"]
    settled = df["paid_share"] >= 0.999
    df["days_late"] = (df["last_payment"].where(settled, as_of) - df["due_date"]).dt.days
    return df.reset_index()


def _reason(z_row):
    extreme = z_row[z_row.abs() >= REASON_Z].abs().sort_values(ascending=False)
    if extreme.empty:
        return "unusual combination"
    return ", ".join(
        f"{'high' if z_row[col] > 0 else 'low'} {FEATURES[col]}" for col in extreme.index
    )


def detect(features, contamination):
    """Flag the most isolated invoices. `contamination` is the expected share of anomalies."""
    if not 0 < contamination <= 0.5:
        raise ValueError("contamination must be in (0, 0.5]")
    scaled = StandardScaler().fit_transform(features[list(FEATURES)])
    forest = IsolationForest(n_estimators=300, contamination=contamination, random_state=42).fit(scaled)

    out = features.copy()
    out["anomaly_score"] = -forest.score_samples(scaled)
    out["is_anomaly"] = forest.predict(scaled) == -1
    z = pd.DataFrame(scaled, columns=list(FEATURES), index=out.index)
    out["reason"] = [_reason(row) for _, row in z.iterrows()]
    return out


def monthly_trend(result):
    """Number and share of flagged invoices per month."""
    trend = result.set_index("invoice_date").resample("ME").agg(
        {"invoice_id": "count", "is_anomaly": "sum"}
    )
    trend.columns = ["invoices", "anomalies"]
    trend["rate"] = trend["anomalies"] / trend["invoices"].where(trend["invoices"] > 0)
    return trend
