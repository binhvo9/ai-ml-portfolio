import argparse
from datetime import datetime, timezone
import httpx
import pandas as pd
from pathlib import Path
from .save_utils import save_csv
from src.validate.schema import PriceSchema

COINGECKO = "https://api.coingecko.com/api/v3"

def _to_dt(ms):
    # CoinGecko trả millis
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc)

def fetch_market_chart(symbol_id: str, vs: str, days: int) -> pd.DataFrame:
    url = f"{COINGECKO}/coins/{symbol_id}/market_chart"
    params = {"vs_currency": vs, "days": days}
    with httpx.Client(timeout=30) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

    # data["prices"] = [[ts_ms, price], ...]
    prices = pd.DataFrame(data.get("prices", []), columns=["ts_ms", "price"])
    mktcap = pd.DataFrame(data.get("market_caps", []), columns=["ts_ms", "market_cap"])
    vol    = pd.DataFrame(data.get("total_volumes", []), columns=["ts_ms", "volume"])

    df = prices.merge(mktcap, on="ts_ms", how="left").merge(vol, on="ts_ms", how="left")
    df["ts"] = df["ts_ms"].apply(_to_dt)
    df = df.drop(columns=["ts_ms"])

    # sort & basic cleaning
    df = df.sort_values("ts").drop_duplicates(subset=["ts"])
    # loại giá <= 0
    df = df[df["price"] > 0].copy()

    # gán meta
    df["symbol"] = symbol_id
    # CoinGecko market_chart daily -> freq 'hourly' nếu days<=90; >90 thường daily
    freq = "hourly" if days <= 90 else "daily"
    df["freq"] = freq

    # Nếu daily: resample về 'D' cho ổn định (fill nhẹ)
    if freq == "daily":
        df = (df.set_index("ts")
                .resample("D")
                .agg({"price":"last", "market_cap":"last", "volume":"sum"})
                .ffill()
                .reset_index())
        df["symbol"] = symbol_id
        df["freq"] = "daily"

    # validate schema
    PriceSchema.validate(df)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="bitcoin", help="coingecko id: bitcoin, ethereum, ...")
    ap.add_argument("--vs", default="usd")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--outdir_raw", default="data/raw")
    ap.add_argument("--outdir_interim", default="data/interim")
    args = ap.parse_args()

    df = fetch_market_chart(args.symbol, args.vs, args.days)

    raw_path = Path(args.outdir_raw) / f"{args.symbol}_{args.vs}_{args.days}d.csv"
    interim_path = Path(args.outdir_interim) / f"{args.symbol}_{args.vs}_clean.csv"

    save_csv(df, str(raw_path))
    # Với Level-2, raw và interim có thể giống lúc đầu; sau này thêm bước clean nâng cao.
    save_csv(df, str(interim_path))

    print(f"[OK] Saved RAW -> {raw_path}")
    print(f"[OK] Saved CLEAN -> {interim_path}")

if __name__ == "__main__":
    main()
