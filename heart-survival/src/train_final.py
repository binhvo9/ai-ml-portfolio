# src/train_final.py
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) load full data
    df = pd.read_csv("data/breast_cancer.csv")
    X = df.drop("target", axis=1)
    y = df["target"].astype(int)

    # 2) split (same seed as before để so sánh apples-to-apples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3) define pipeline (scaler + logistic)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, n_jobs=None, class_weight="balanced"))
    ])

    # 4) train
    pipe.fit(X_train, y_train)

    # 5) eval
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # 6) save artifacts
    joblib.dump(pipe, ARTIFACTS_DIR / "model.joblib")
    json.dump(list(X.columns), open(ARTIFACTS_DIR / "columns.json", "w"))
    json.dump(metrics, open(ARTIFACTS_DIR / "metrics.json", "w"), indent=2)

    print("✅ Saved:")
    print(" - artifacts/model.joblib")
    print(" - artifacts/columns.json")
    print(" - artifacts/metrics.json")
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
