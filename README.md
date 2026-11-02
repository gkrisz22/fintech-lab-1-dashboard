# Decision Support Dashboard

Streamlit dashboard for a financial manager built on the ERP data used in the risk management task (2024 and 2025).

## Setup

Requires Python 3.11 or newer. Tested with Python 3.14, Streamlit 1.66, pandas 3.1 and scikit-learn 1.9.

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Structure

* `app.py` is the Streamlit layout (filters, tabs, charts, explanations).
* `src/data.py` loads `data/financial_analytics.db` and applies the filters.
* `src/business.py`, `src/risk.py`, `src/predictive.py` and `src/anomaly.py` do the calculations.
* `src/charts.py` builds the Plotly figures.
