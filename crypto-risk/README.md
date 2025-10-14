# Crypto Risk & Price Intelligence (Python)

> **Goal:** A real **AI engineering** capstone in Python that
>
> 1. pulls **live crypto prices** (Binance, free),
> 2. ingests **messy historical data**,
> 3. builds **features** + **walk-forward CV** with **LightGBM + Optuna**,
> 4. **serves predictions** via **FastAPI** (no JS stack).

---

## ✨ Features

* **Server-side API** calls (Binance) with retry/backoff
* **Structured project**: ingest → validate → features → train → serve
* **Time-series CV** (leak-safe), LightGBM, Optuna tuning
* **FastAPI app**: `/` live prices, `/predict` risk prob, `/health`, `/metrics`

---

## 🧱 Tech Stack

* **Web**: FastAPI + Jinja2
* **HTTP client**: httpx
* **Data**: pandas, pandera (schema)
* **ML**: scikit-learn, lightgbm, optuna
* **Viz**: matplotlib/plotly (optional)
* **Versioning**: Git/GitHub

---

## 📁 Project Structure

```
crypto-risk/
  README.md
  requirements.txt
  configs/
    base.yaml
  data/
    raw/        # pulled CSV
    interim/    # cleaned CSV
    processed/  # features CSV
  reports/
    metrics.json
  src/
    __init__.py
    app/
      __init__.py
      main.py
      templates/
        home.html
        predict.html
      services/
        __init__.py
        binance.py
        inference.py
    features/
      __init__.py
      build_features.py
    ingest/
      fetch_history_binance.py
      save_utils.py
    models/
      __init__.py
      train.py
      artifacts/
        lgbm_drop2pct.joblib
        features.json
```

---

## ⚡ Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run app
uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
# Home:    http://127.0.0.1:8000/
# Predict: http://127.0.0.1:8000/predict
# Health:  http://127.0.0.1:8000/health
# Metrics: http://127.0.0.1:8000/metrics
```

---

## 🔄 Data Ingestion (Binance, free)

Pull ~730 days of daily candles (no API key):

```bash
python -m src.ingest.fetch_history_binance --symbol BTCUSDT --interval 1d --days 730
python -m src.ingest.fetch_history_binance --symbol ETHUSDT --interval 1d --days 730
```

**Output files**

* `data/raw/<SYMBOL>_1d_730d.csv`
* `data/interim/<SYMBOL>_1d_clean.csv`

Columns: `ts, price, market_cap(NA), volume, symbol, freq`

---

## 🧩 Feature Engineering

Build lags, returns, rolling stats, RSI, MACD, volatility:

```bash
python -m src.features.build_features \
  --infile data/interim/BTCUSDT_1d_clean.csv \
  --outfile data/processed/BTCUSDT_1d_features.csv

python -m src.features.build_features \
  --infile data/interim/ETHUSDT_1d_clean.csv \
  --outfile data/processed/ETHUSDT_1d_features.csv
```

**Target:** `y_drop2pct = 1` if next-day return ≤ −2%
**NaN handling:** drops only on **feature columns** (keeps rows with NA market_cap).

---

## 🏋️ Train + Tune (LightGBM + Optuna)

```bash
mkdir -p src/models/artifacts reports
python -m src.models.train \
  --infile data/processed/BTCUSDT_1d_features.csv \
  --n_trials 40 --n_splits 5
```

**Artifacts**

* `src/models/artifacts/lgbm_drop2pct.joblib`
* `src/models/artifacts/features.json`
* `reports/metrics.json` (metrics)

---

## 🚀 Run the App

```bash
uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
# /           → live BTC/ETH prices (Binance)
# /predict    → probability of ≥2% drop tomorrow (BTC)
# /health     → status
# /metrics    → saved metrics (if any)
```

---

## ⚙️ Config (optional)

`configs/base.yaml` (example):

```yaml
symbol: "bitcoin"
vs_currency: "usd"
history_days: 730
freq: "daily"
target:
  type: "binary_drop"
  drop_threshold_pct: 2.0
cv:
  n_splits: 5
optuna:
  n_trials: 40
```

---

## 🧪 Smoke Tests

* Home loads prices → `http://127.0.0.1:8000/`
* `data/processed/BTCUSDT_1d_features.csv` has **rows > 500**
* Model artifacts exist in `src/models/artifacts/`
* `/predict` shows a probability in `[0,1]`

---

## 🛠️ Troubleshooting

* **127.0.0.1 timeout** → run with explicit host/port:
  `uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload`
* **ImportError** → ensure packages are packages:
  `touch src/__init__.py src/app/__init__.py src/app/services/__init__.py src/features/__init__.py src/models/__init__.py`
* **KeyError: feature not in index** → artifacts expect a feature not present at inference.
  This repo **auto-aligns** inference features (missing → 0.0, coerces numeric).
* **Empty training data** → rebuild features; ensure `clean_final` only drops NaN on feature columns.

---

## 📌 Roadmap

* [ ] Threshold selection & calibration
* [ ] Explainability `/explain` (feature importance, SHAP)
* [ ] Multi-asset select on UI (BTC/ETH)
* [ ] Prefect flow (daily ingest→features→train)
* [ ] Cloud deploy (Render/Fly/Railway) *(no Docker section included here)*

---

## 📝 License

