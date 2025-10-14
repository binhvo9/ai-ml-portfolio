import json
import joblib
import pandas as pd

def load_artifacts(
    path_model: str = "src/models/artifacts/lgbm_drop2pct.joblib",
    path_feats: str = "src/models/artifacts/features.json",
):
    """Load trained model + list of feature columns."""
    model = joblib.load(path_model)
    with open(path_feats) as f:
        feat_cols = json.load(f)
    return model, feat_cols

def _align_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """
    Ensure df has exactly the training features:
    - add missing columns filled with 0.0
    - cast to numeric
    - drop rows with any NaN in feature columns
    """
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").dropna()
    return X

def predict_from_dataframe(model, feat_cols: list[str], df_features: pd.DataFrame) -> float | None:
    """Return probability using the most recent valid row; None if not enough data."""
    if df_features is None or df_features.empty:
        return None
    df = df_features.sort_values("ts")
    X = _align_features(df, feat_cols)
    if X.empty:
        return None
    proba = model.predict_proba(X.iloc[[-1]])[:, 1][0]
    return float(proba)
