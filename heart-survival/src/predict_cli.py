# src/predict_cli.py
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"
COLUMNS_PATH = ARTIFACTS / "columns.json"

def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing artifacts/model.joblib — run train_final.py first.")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError("Missing artifacts/columns.json — run train_final.py first.")
    model = joblib.load(MODEL_PATH)
    columns = json.load(open(COLUMNS_PATH))
    return model, columns

def show_columns(columns):
    print("=== Required columns (order) ===")
    for i, c in enumerate(columns):
        print(f"{i+1:02d}. {c}")

def predict_row(model, columns, values):
    arr = np.array(values, dtype=float).reshape(1, -1)
    proba = float(model.predict_proba(arr)[0, 1])
    pred = int(proba >= 0.5)
    return pred, proba

def predict_csv(model, columns, csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    # Reorder/select columns to match training schema
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    X = df[columns].astype(float)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["pred"] = pred
    out["prob"] = proba
    if out_path:
        out.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")
    else:
        print(out.head(10))
    return out

def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Survival Prediction CLI")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--show-columns", action="store_true", help="Show required column order")
    g.add_argument("--values", type=str, help="Comma-separated values in required column order")
    g.add_argument("--csv", type=str, help="Path to CSV containing header with required columns")
    parser.add_argument("--out", type=str, help="Output CSV path (for --csv mode)")
    args = parser.parse_args()

    model, columns = load_artifacts()

    if args.show_columns:
        show_columns(columns)
        return

    if args.values:
        vals = [v.strip() for v in args.values.split(",")]
        if len(vals) != len(columns):
            raise ValueError(f"Expected {len(columns)} values, got {len(vals)}. Use --show-columns.")
        pred, proba = predict_row(model, columns, vals)
        print("Prediction:", pred, "(1=benign, 0=malignant)")  # theo sklearn dataset: 1=benign, 0=malignant
        print("Probability of class 1 (benign):", round(proba, 4))
        return

    if args.csv:
        predict_csv(model, columns, args.csv, args.out)

if __name__ == "__main__":
    main()
