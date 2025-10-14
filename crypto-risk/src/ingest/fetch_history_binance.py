# src/ingest/fetch_history_binance.py
import argparse
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import pandas as pd

from src.ingest.save_utils import save_csv
from src.validate.schema import PriceSchema

BASE = "https://api.binance.com"

def _ms(dt): return int(dt.timestamp() * 1000)

def fetch_klines(symbol="BTCUSDT", interval="1d", days=730):
    """Fetch klines over the past `days` for symbol/interval (no API key)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 2)  # buffer
    limit = 1000

    rows = []
    with httpx.Client(timeout=30) as client:
        cur_start = start
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": _ms(cur_start),
                "endTime": _ms(end),
                "limit": limit
            }
            r = client.get(f"{BASE}/api/v3/klines", params=params)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            # next window: last kline close time + 1ms
            next_ms = batch[-1][6] + 1
            cur_start = datetime.fromtimestamp(next_ms/1000, tz=timezone.utc)
            if cur_start >= end or len(batch) < limit:
                break

    if not rows:
        raise RuntimeError("No klines returned.")

    # Binance kline schema:
    # 0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume,
    # 6 close time, 7 quote asset volume, ...
    # ... imports giữ nguyên
    import numpy as np

    # ... phần fetch_klines không đổi tới đoạn tạo df

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
    ])
    df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    # LẤY NHỮNG CỘT CẦN THIẾT; BỎ LUÔN quote_vol ĐỂ KHỎI VƯỚNG STRICT
    df = df[["ts", "close", "volume"]].copy()

    # ép kiểu
    df["price"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df.drop(columns=["close"], inplace=True)

    # market_cap không có từ Binance → để NaN float
    df["market_cap"] = np.nan  # float NaN

    df["symbol"] = symbol
    df["freq"] = "daily" if interval.endswith("d") else "intraday"

    # sort, dedupe, ensure positive price
    df = (df.sort_values("ts")
          .drop_duplicates(subset=["ts"])
          .query("price > 0")
          .reset_index(drop=True))

    # VALIDATE
    PriceSchema.validate(df)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1d", help="e.g. 1d, 4h, 1h")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--outdir_raw", default="data/raw")
    ap.add_argument("--outdir_interim", default="data/interim")
    args = ap.parse_args()

    df = fetch_klines(args.symbol, args.interval, args.days)

    raw_path = Path(args.outdir_raw) / f"{args.symbol}_{args.interval}_{args.days}d.csv"
    interim_path = Path(args.outdir_interim) / f"{args.symbol}_{args.interval}_clean.csv"

    save_csv(df, str(raw_path))
    save_csv(df, str(interim_path))
    print(f"[OK] Saved RAW -> {raw_path}")
    print(f"[OK] Saved CLEAN -> {interim_path}")

if __name__ == "__main__":
    main()
