import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
import joblib

RANDOM_SEED = 42

def split_Xy(df):
    y = df["y_drop2pct"].values
    drop_cols = ["ts","symbol","freq","next_price","next_ret","y_drop2pct"]
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].values
    return X, y, feat_cols

def cross_val_objective(df, n_splits=5):
    X, y, _ = split_Xy(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_SEED,
            "class_weight": "balanced"
        }
        ap_scores, roc_scores = [], []
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xv = X[train_idx], X[val_idx]
            ytr, yv = y[train_idx], y[val_idx]
            clf = LGBMClassifier(**params)
            clf.fit(Xtr, ytr)
            pv = clf.predict_proba(Xv)[:,1]
            ap_scores.append(average_precision_score(yv, pv))   # PR-AUC
            roc_scores.append(roc_auc_score(yv, pv))            # ROC-AUC
        # tối ưu PR-AUC (nhạy với class imbalance)
        return np.mean(ap_scores)
    return objective

def fit_full(df, best_params):
    X, y, feat_cols = split_Xy(df)
    model = LGBMClassifier(**best_params, random_state=RANDOM_SEED, class_weight="balanced")
    model.fit(X, y)
    return model, feat_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--artifact_dir", default="src/models/artifacts")
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.infile, parse_dates=["ts"]).sort_values("ts")

    objective = cross_val_objective(df, n_splits=args.n_splits)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best_params = study.best_trial.params
    print("[BEST]", best_params)

    model, feat_cols = fit_full(df, best_params)
    joblib.dump(model, f"{args.artifact_dir}/lgbm_drop2pct.joblib")
    with open(f"{args.artifact_dir}/features.json","w") as f:
        json.dump(feat_cols, f)

    # save metrics on whole set (diagnostic)
    pv = model.predict_proba(df[feat_cols].values)[:,1]
    metrics = {
        "pr_auc": float(average_precision_score(df["y_drop2pct"], pv)),
        "roc_auc": float(roc_auc_score(df["y_drop2pct"], pv))
    }
    with open("reports/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("[OK] Saved model & features. Metrics:", metrics)

if __name__ == "__main__":
    main()
