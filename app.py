"""Decision Support Dashboard for a financial manager.

Run with: streamlit run app.py
"""

import streamlit as st

from src import anomaly, business, charts, data, predictive, risk

st.set_page_config(page_title="Decision Support Dashboard", layout="wide")


@st.cache_data
def get_tables():
    return data.load_tables()


@st.cache_data
def get_anomalies(contamination):
    tables = get_tables()
    features = anomaly.invoice_features(tables["invoices"], tables["sales"], tables["payments"])
    return anomaly.detect(features, contamination)


@st.cache_data
def get_prediction(weekly, model_name, test_weeks, horizon):
    result, metrics, error_std = predictive.evaluate(weekly, model_name, test_weeks)
    fc = predictive.forecast(weekly, model_name, horizon, error_std)
    return result, metrics, fc


def money(value):
    return f"{value:,.0f}"


def business_section(sales_dim, sales, start, end, data_start):
    k = business.kpis(sales_dim, start, end, data_start)
    cols = st.columns(5)
    cols[0].metric("Total revenue", money(k["revenue"]))
    prev_start, prev_end = k["previous_period"]
    cols[1].metric(
        "Revenue growth",
        "n/a" if k["growth"] is None else f"{k['growth']:+.1%}",
        help=f"Compared with the previous period of the same length ({prev_start:%Y-%m-%d} to {prev_end:%Y-%m-%d}). "
        "Shown as n/a when that period is outside the data.",
    )
    cols[2].metric("Customers", f"{k['customers']:,}")
    cols[3].metric("Transactions", f"{k['transactions']:,}", help="Number of invoices")
    cols[4].metric("Average transaction value", money(k["avg_transaction"]))
    st.caption("Revenue is the net sales value after discount and before tax. A transaction is one invoice.")

    st.plotly_chart(charts.monthly_revenue(business.monthly_revenue(sales)))
    st.caption("Monthly revenue with a 3 month moving average. The average line shows the trend without single strong or weak months.")

    n = st.slider("Number of products and customers to show", min_value=3, max_value=20, value=10)
    left, right = st.columns(2)
    left.plotly_chart(charts.top_bar(business.top_products(sales, n), "product_name", f"Top {n} products by revenue"))
    right.plotly_chart(charts.top_bar(business.top_customers(sales, n), "customer_name", f"Top {n} customers by revenue", charts.ORANGE))
    st.caption("These products and customers bring the most revenue. A high share from a few customers means a higher dependency risk.")

    st.plotly_chart(charts.category_revenue(business.category_revenue(sales)))
    st.caption("Revenue and profit by product category show where the margin comes from.")


def risk_section(sales, start, end):
    weekly = risk.weekly_series(sales, start, end)
    returns = risk.weekly_returns(weekly)
    if len(returns) < 8:
        st.warning("Risk analysis needs at least 8 complete weeks with sales. Widen the date range or the filters.")
        return

    st.subheader("Volatility")
    max_window = min(26, len(returns) - 1)
    window = st.slider("Volatility window (weeks)", min_value=4, max_value=max_window, value=min(12, max_window))
    vol = risk.rolling_volatility(returns, window)
    cols = st.columns(3)
    cols[0].metric("Latest volatility", f"{vol.iloc[-1]:.1%}")
    cols[1].metric("Average volatility", f"{vol.mean():.1%}")
    cols[2].metric("Highest volatility", f"{vol.max():.1%}", help=f"Week ending {vol.idxmax():%Y-%m-%d}")
    st.plotly_chart(charts.volatility(vol, window))
    st.caption(
        "Volatility is the standard deviation of the week over week change in revenue. "
        "Higher values mean less predictable cash inflows."
    )

    st.subheader("Portfolio performance")
    cat_weekly = risk.weekly_by_category(sales, start, end)
    if cat_weekly.shape[1] < 2:
        st.info("Select at least two product categories to analyze them as a portfolio.")
    else:
        div = risk.diversification(cat_weekly)
        cols = st.columns(3)
        cols[0].metric("Total weekly profit (mean)", money(cat_weekly.sum(axis=1).mean()))
        cols[1].metric("Portfolio risk (std)", money(div["portfolio_std"]), help="Std of total weekly profit")
        cols[2].metric("Risk reduction from diversification", f"{div['reduction']:.0%}", help=f"Compared with the sum of category risks ({money(div['sum_std'])})")
        st.plotly_chart(charts.cumulative_profit(cat_weekly))
        left, right = st.columns(2)
        left.plotly_chart(charts.risk_return(risk.risk_return(cat_weekly)))
        right.plotly_chart(charts.correlation(cat_weekly))
        st.caption(
            "Each category is treated as an asset with its weekly profit as return. "
            "Categories higher and further left give more profit per unit of risk. "
            "Low correlations mean the categories rarely have bad weeks together, so the total risk is lower than the sum of the parts."
        )

    st.subheader("Value at Risk (VaR)")
    confidence = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95, format_func=lambda c: f"{c:.0%}")
    var_table = risk.value_at_risk(returns, confidence, weekly.mean())
    cols = st.columns(2)
    cols[0].metric("Historical VaR", f"{var_table.loc['Historical', 'VaR (return)']:.1%}", money(var_table.loc["Historical", "VaR (revenue)"]), delta_color="off")
    cols[1].metric("Parametric VaR", f"{var_table.loc['Parametric', 'VaR (return)']:.1%}", money(var_table.loc["Parametric", "VaR (revenue)"]), delta_color="off")
    st.plotly_chart(charts.var_histogram(returns, var_table, confidence))
    st.caption(
        f"In {confidence:.0%} of the weeks revenue does not fall more than the VaR compared with the previous week. "
        "The amount is the drop applied to the average weekly revenue. "
        "The historical method uses the observed returns, the parametric one assumes a normal distribution."
    )


def predictive_section(sales, start, end):
    weekly = risk.weekly_series(sales, start, end)
    cols = st.columns(3)
    model_name = cols[0].selectbox("Model", predictive.MODEL_NAMES)
    test_weeks = cols[1].slider("Test weeks", min_value=8, max_value=26, value=16)
    horizon = cols[2].slider("Forecast horizon (weeks)", min_value=1, max_value=12, value=8)

    try:
        result, metrics, fc = get_prediction(weekly, model_name, test_weeks, horizon)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.subheader("Revenue forecast")
    cols = st.columns(2)
    cols[0].metric("Next week forecast", money(fc["forecast"].iloc[0]))
    cols[1].metric(f"Total for the next {horizon} weeks", money(fc["forecast"].sum()))
    st.plotly_chart(charts.forecast(weekly, fc))
    st.caption(
        "The model predicts weekly revenue from the last four weeks and the month. "
        "The shaded band is the range where 95% of the weeks should fall, based on the test errors."
    )

    st.subheader("ML predictions")
    st.plotly_chart(charts.predictions(result, model_name))
    st.caption(
        f"The model was trained on the weeks before the last {test_weeks} and then predicted these unseen weeks. "
        "The naive baseline simply repeats last week's revenue."
    )

    st.subheader("Model performance")
    st.dataframe(
        metrics,
        column_config={
            "MAE": st.column_config.NumberColumn(format="localized"),
            "RMSE": st.column_config.NumberColumn(format="localized"),
            "MAPE": st.column_config.NumberColumn(format="percent"),
            "R2": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    model_mae, naive_mae = metrics["MAE"].iloc[0], metrics["MAE"].iloc[1]
    if model_mae < naive_mae:
        verdict = f"The model's average error is {1 - model_mae / naive_mae:.0%} lower than the naive baseline."
    else:
        verdict = "The model does not beat the naive baseline here, so use the forecast only as a rough level estimate."
    st.caption(
        "MAE and RMSE are the average errors in revenue, MAPE is the average error in percent and "
        f"R2 shows how much of the weekly variation the model explains. {verdict}"
    )


def anomaly_section(sales):
    pct = st.slider("Expected share of anomalies (%)", min_value=1, max_value=10, value=5)
    result = get_anomalies(pct / 100)
    view = result[result["invoice_id"].isin(sales["invoice_id"].unique())]
    flagged = view[view["is_anomaly"]].sort_values("anomaly_score", ascending=False)

    cols = st.columns(3)
    cols[0].metric("Detected anomalies", f"{len(flagged):,}")
    cols[1].metric("Share of transactions", f"{len(flagged) / len(view):.1%}" if len(view) else "n/a")
    cols[2].metric("Value of flagged invoices", money(flagged["invoice_value"].sum()))
    st.caption(
        "An Isolation Forest compares every invoice on value, size, discount and payment behaviour and flags the ones that differ most. "
        "Open invoices count as late until the last payment date in the data, so old unpaid invoices stand out most."
    )

    st.subheader("Suspicious transactions")
    if flagged.empty:
        st.info("No anomalies in the current selection.")
    else:
        st.dataframe(
            flagged[["invoice_id", "invoice_date", "customer_name", "country", "invoice_value", "paid_share", "days_late", "anomaly_score", "reason"]],
            hide_index=True,
            column_config={
                "invoice_id": "Invoice",
                "invoice_date": st.column_config.DateColumn("Date"),
                "customer_name": "Customer",
                "country": "Country",
                "invoice_value": st.column_config.NumberColumn("Invoice value", format="localized"),
                "paid_share": st.column_config.NumberColumn("Paid", format="percent"),
                "days_late": st.column_config.NumberColumn("Days late"),
                "anomaly_score": st.column_config.NumberColumn("Score", format="%.3f"),
                "reason": "Reason",
            },
        )
        st.caption("Sorted by anomaly score, higher means more unusual. The reason lists the features that are far from typical.")

    st.subheader("Anomaly trends")
    st.plotly_chart(charts.anomaly_trend(anomaly.monthly_trend(view)))
    st.plotly_chart(charts.anomaly_scatter(view))
    st.caption("Red points are flagged invoices. Most are unpaid or paid very late, which points to collection risk.")


st.title("Decision Support Dashboard")
st.caption(
    "Sales, invoices and payments from the company ERP for 2024 and 2025. "
    "Use the filters on the left to focus on a period, product category or country."
)

try:
    tables = get_tables()
except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.stop()

all_sales = tables["sales"]
data_start, data_end = all_sales["sale_date"].min(), all_sales["sale_date"].max()

with st.sidebar:
    st.header("Filters")
    date_value = st.date_input(
        "Date range",
        value=(data_start.date(), data_end.date()),
        min_value=data_start.date(),
        max_value=data_end.date(),
    )
    category_options = sorted(all_sales["category"].unique())
    categories = st.multiselect("Product categories", category_options, default=category_options)
    country_options = sorted(all_sales["country"].unique())
    countries = st.multiselect("Customer countries", country_options, default=country_options)

try:
    start, end = data.parse_date_range(date_value)
    data.check_selection(categories, "product category")
    data.check_selection(countries, "country")
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

sales_dim = data.filter_dimensions(all_sales, categories, countries)
sales = data.filter_dates(sales_dim, "sale_date", start, end)
if sales.empty:
    st.warning("No sales match the current filters. Change the date range, categories or countries.")
    st.stop()

tab_business, tab_risk, tab_predictive, tab_anomaly = st.tabs(
    ["Business Performance", "Risk Analytics", "Predictive Analytics", "Anomaly Detection"]
)
with tab_business:
    business_section(sales_dim, sales, start, end, data_start)
with tab_risk:
    risk_section(sales, start, end)
with tab_predictive:
    predictive_section(sales, start, end)
with tab_anomaly:
    anomaly_section(sales)
