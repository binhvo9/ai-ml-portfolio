import logging
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import httpx
import pandas as pd

from .services.binance import get_simple_prices
from .services.inference import load_artifacts, predict_from_dataframe
from ..features.build_features import (
    add_returns, add_roll_stats, add_volatility,
    add_rsi, add_macd, add_lags, build_targets, clean_final
)

# ====== APP & LOGGING ======
app = FastAPI(title="Crypto Risk & Price Intelligence (Python)")
templates = Jinja2Templates(directory="src/app/templates")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("app")

BINANCE = "https://api.binance.com"

# ====== HELPERS ======
async def fetch_recent_klines(symbol: str = "BTCUSDT", interval: str = "1d", days: int = 120) -> pd.DataFrame:
    """Fetch last `days` klines from Binance (no API key)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 2)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{BINANCE}/api/v3/klines", params=params)
        r.raise_for_status()
        rows = r.json()

    # Binance kline columns
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume","close_time",
        "quote_vol","trades","taker_base","taker_quote","ignore"
    ])
    # standardize
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df[["ts", "close", "volume"]].rename(columns={"close": "price"})
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["symbol"] = symbol
    df["freq"] = "daily"
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    return df

# ====== ROUTES ======
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    prices = await get_simple_prices(("BTCUSDT", "ETHUSDT"))
    return templates.TemplateResponse("home.html", {"request": request, "prices": prices})

@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    log.info("Serving /predict")
    try:
        # 1) artifacts
        model, feat_cols = load_artifacts()

        # 2) recent data
        df = await fetch_recent_klines("BTCUSDT", "1d", 120)
        if df.empty:
            return templates.TemplateResponse("predict.html", {"request": request, "error": "No recent data."})

        # 3) build features (MUST mirror training)
        df_feat = df.copy()
        df_feat = add_returns(df_feat)
        df_feat = add_roll_stats(df_feat)
        df_feat = add_volatility(df_feat)
        df_feat = add_rsi(df_feat)
        df_feat = add_macd(df_feat)
        df_feat = add_lags(df_feat)
        df_feat = build_targets(df_feat)         # target not used for inference
        df_feat = clean_final(df_feat)           # drop NaN on features only

        # 4) predict
        proba = predict_from_dataframe(model, feat_cols, df_feat)
        if proba is None:
            return templates.TemplateResponse(
                "predict.html", {"request": request, "error": "Not enough valid feature rows after alignment."}
            )
        result = {"prob_drop_2pct_tomorrow": round(proba, 4)}
        return templates.TemplateResponse("predict.html", {"request": request, "result": result})

    except FileNotFoundError as e:
        return templates.TemplateResponse("predict.html", {"request": request, "error": f"Artifact missing: {e}"})
    except Exception as e:
        log.exception("Predict error")
        return templates.TemplateResponse("predict.html", {"request": request, "error": f"{type(e).__name__}: {e}"})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    import os, json
    p = "reports/metrics.json"
    if os.path.exists(p):
        return json.load(open(p))
    return {"info": "no metrics yet"}
