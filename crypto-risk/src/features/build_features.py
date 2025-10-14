import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def add_returns(df, col="price"):
    df["ret_1d"] = df[col].pct_change()
    for w in [3,7,14,30]:
        df[f"ret_{w}d"] = df[col].pct_change(w)
    return df

def add_roll_stats(df, col="price"):
    for w in [3,7,14,30]:
        m = max(2, w//2)
        df[f"roll_mean_{w}"] = df[col].rolling(w, min_periods=m).mean()
        df[f"roll_std_{w}"]  = df[col].rolling(w, min_periods=m).std()
        df[f"roll_min_{w}"]  = df[col].rolling(w, min_periods=m).min()
        df[f"roll_max_{w}"]  = df[col].rolling(w, min_periods=m).max()
    return df

def add_volatility(df):
    # Parkinson volatility (approx) từ high/low không có => dùng std của returns
    df["vol_7"]  = df["ret_1d"].rolling(7).std()
    df["vol_14"] = df["ret_1d"].rolling(14).std()
    df["vol_30"] = df["ret_1d"].rolling(30).std()
    return df

def add_rsi(df, col="price", period=14):
    delta = df[col].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=5).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, col="price"):
    ema12 = df[col].ewm(span=12, adjust=False).mean()
    ema26 = df[col].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def add_lags(df, col="price"):
    for l in [1,2,3,7,14,30]:
        df[f"{col}_lag{l}"] = df[col].shift(l)
    return df

def build_targets(df, drop_threshold_pct=2.0):
    # Binary: 1 nếu ngày mai giảm <= -2%
    df["next_price"] = df["price"].shift(-1)
    df["next_ret"] = (df["next_price"] - df["price"]) / df["price"]
    df["y_drop2pct"] = (df["next_ret"] <= -(drop_threshold_pct/100.0)).astype(int)
    return df

def clean_final(df):
    # chỉ drop NaN trên các cột feature, tránh market_cap/ts/... làm rớt hết
    drop_cols = ["ts","symbol","freq","next_price","next_ret","y_drop2pct","market_cap"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    return df.dropna(subset=feat_cols).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="data/interim/BTCUSDT_1d_clean.csv")
    ap.add_argument("--outfile", required=True, help="data/processed/BTCUSDT_1d_features.csv")
    ap.add_argument("--drop_threshold_pct", type=float, default=2.0)
    args = ap.parse_args()

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.infile, parse_dates=["ts"])
    df = df.sort_values("ts")

    df = add_returns(df)
    df = add_roll_stats(df)
    df = add_volatility(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_lags(df)
    df = build_targets(df, drop_threshold_pct=args.drop_threshold_pct)
    df = clean_final(df)

    df.to_csv(args.outfile, index=False)
    print(f"[OK] Features saved -> {args.outfile}")

if __name__ == "__main__":
    main()
