"""Plotly figures for the dashboard. Each function takes prepared data and returns a figure."""

import plotly.express as px
import plotly.graph_objects as go

BLUE = "#2563eb"
ORANGE = "#ea580c"
GREY = "#9ca3af"
RED = "#dc2626"
CATEGORY_COLORS = px.colors.qualitative.Safe


def _layout(fig, title, x_title=None, y_title=None, height=380):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    return fig


def monthly_revenue(monthly):
    fig = go.Figure()
    fig.add_bar(x=monthly.index, y=monthly["revenue"], name="Monthly revenue", marker_color=BLUE)
    fig.add_scatter(x=monthly.index, y=monthly["ma3"], name="3 month moving average", line=dict(color=ORANGE, width=3))
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Revenue over time", "Month", "Revenue")


def top_bar(table, label_col, title, color=BLUE):
    table = table.sort_values("revenue")
    fig = px.bar(table, x="revenue", y=label_col, orientation="h", hover_data=table.columns)
    fig.update_traces(marker_color=color)
    fig.update_xaxes(tickformat=",.0f")
    return _layout(fig, title, "Revenue", None, height=max(300, 32 * len(table) + 80))


def category_revenue(table):
    fig = go.Figure()
    fig.add_bar(x=table["category"], y=table["revenue"], name="Revenue", marker_color=BLUE)
    fig.add_bar(x=table["category"], y=table["profit"], name="Profit", marker_color=ORANGE)
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Revenue and profit by category", None, "Amount")


def volatility(vol, window):
    fig = go.Figure()
    fig.add_scatter(x=vol.index, y=vol, name="Rolling volatility", line=dict(color=BLUE, width=2))
    fig.add_hline(y=vol.mean(), line_dash="dash", line_color=GREY, annotation_text="average")
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, f"{window} week rolling volatility of weekly revenue returns", "Week", "Volatility")


def cumulative_profit(cat_weekly):
    cumulative = cat_weekly.cumsum()
    fig = px.line(cumulative, color_discrete_sequence=CATEGORY_COLORS)
    fig.update_yaxes(tickformat=",.0f")
    fig.update_layout(legend_title_text="")
    return _layout(fig, "Cumulative profit by category", "Week", "Profit")


def risk_return(table):
    fig = px.scatter(
        table.reset_index(),
        x="std_profit",
        y="mean_profit",
        text="category",
        color="category",
        color_discrete_sequence=CATEGORY_COLORS,
    )
    fig.update_traces(marker_size=14, textposition="top center", showlegend=False)
    fig.update_xaxes(tickformat=",.0f")
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Risk vs return by category", "Risk (std of weekly profit)", "Return (mean weekly profit)")


def correlation(cat_weekly):
    corr = cat_weekly.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    return _layout(fig, "Correlation of weekly profit between categories")


def var_histogram(returns, var_table, confidence):
    fig = px.histogram(returns, nbins=30, color_discrete_sequence=[BLUE])
    fig.update_layout(showlegend=False)
    for name, color in [("Historical", RED), ("Parametric", ORANGE)]:
        value = var_table.loc[name, "VaR (return)"]
        fig.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=f"{name} VaR")
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, f"Weekly revenue returns and {confidence:.0%} VaR", "Weekly return", "Number of weeks")


def forecast(weekly, fc):
    fig = go.Figure()
    fig.add_scatter(x=weekly.index, y=weekly, name="Actual weekly revenue", line=dict(color=BLUE))
    fig.add_scatter(
        x=list(fc.index) + list(fc.index[::-1]),
        y=list(fc["upper"]) + list(fc["lower"][::-1]),
        fill="toself",
        fillcolor="rgba(234, 88, 12, 0.15)",
        line=dict(width=0),
        name="95% interval",
        hoverinfo="skip",
    )
    fig.add_scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(color=ORANGE, width=3), mode="lines+markers")
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Weekly revenue forecast", "Week", "Revenue")


def predictions(result, model_name):
    fig = go.Figure()
    fig.add_scatter(x=result.index, y=result["actual"], name="Actual", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=result.index, y=result["predicted"], name=model_name, line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=result.index, y=result["naive"], name="Naive (last week)", line=dict(color=GREY, dash="dot"))
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Predicted vs actual revenue on the test weeks", "Week", "Revenue")


def anomaly_trend(trend):
    fig = go.Figure()
    fig.add_bar(x=trend.index, y=trend["anomalies"], name="Anomalies", marker_color=RED)
    fig.add_scatter(x=trend.index, y=trend["rate"], name="Share of invoices", yaxis="y2", line=dict(color=GREY, width=2))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False, rangemode="tozero"))
    return _layout(fig, "Anomalies per month", "Month", "Number of anomalies")


def anomaly_scatter(result):
    df = result.assign(status=result["is_anomaly"].map({True: "Anomaly", False: "Normal"}))
    fig = px.scatter(
        df,
        x="days_late",
        y="invoice_value",
        color="status",
        color_discrete_map={"Anomaly": RED, "Normal": GREY},
        hover_data=["invoice_id", "customer_name", "paid_share", "reason"],
        opacity=0.7,
    )
    fig.update_layout(legend_title_text="")
    fig.update_yaxes(tickformat=",.0f")
    return _layout(fig, "Invoice value vs days late", "Days late (negative means paid early)", "Invoice value")
