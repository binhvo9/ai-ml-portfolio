# streamlit_app.py
import json
import io
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

ART = Path("artifacts")
MODEL_PATH = ART / "model.joblib"
COLUMNS_PATH = ART / "columns.json"

# ---------- helpers ----------
@st.cache_resource
def load_model_and_columns():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing artifacts/model.joblib — please run src/train_final.py")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError("Missing artifacts/columns.json — please run src/train_final.py")
    model = joblib.load(MODEL_PATH)
    columns = json.load(open(COLUMNS_PATH))
    return model, columns

def ensure_schema(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[columns].astype(float)

def make_template_csv(columns: list[str]) -> bytes:
    tmp = pd.DataFrame(columns=columns, data=[[""] * len(columns)])
    return tmp.to_csv(index=False).encode("utf-8")

def predict_with_threshold(proba: np.ndarray, thr: float) -> np.ndarray:
    return (proba >= thr).astype(int)

def render_confusion_matrix(cm: np.ndarray, labels=("Malignant(0)", "Benign(1)")):
    # simple text grid + dataframe
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm, index=[f"True {l}" for l in labels],
                             columns=[f"Pred {l}" for l in labels]))

def render_roc(y_true: np.ndarray, proba: np.ndarray):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    st.subheader("ROC Curve")
    st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}).set_index("FPR"))
    st.caption(f"ROC AUC = **{auc:.3f}**")

def render_pr(y_true: np.ndarray, proba: np.ndarray):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    st.subheader("Precision–Recall Curve")
    st.line_chart(pd.DataFrame({"Recall": recall, "Precision": precision}).set_index("Recall"))
    st.caption(f"Average Precision = **{ap:.3f}**")

# ---------- app ----------
st.set_page_config(page_title="Survival Prediction (Breast Cancer)", page_icon="🛟", layout="wide")
st.title("🛟 Survival Prediction — Breast Cancer (LogisticRegression Pipeline)")

with st.expander("ℹ️ About this app"):
    st.markdown(
        "- **Model**: StandardScaler → LogisticRegression (`class_weight='balanced'`)\n"
        "- **Target**: 1 = benign, 0 = malignant\n"
        "- **Artifacts**: loaded from `artifacts/`\n"
        "- **Tip**: Use the *Template CSV* to ensure the correct columns & order."
    )

model, columns = load_model_and_columns()

# Sidebar
st.sidebar.header("Settings")
thr = st.sidebar.slider("Decision threshold (class=1 if prob ≥ threshold)", 0.0, 1.0, 0.5, 0.01)
st.sidebar.download_button(
    label="⬇️ Download Template CSV",
    data=make_template_csv(columns),
    file_name="template_input.csv",
    mime="text/csv"
)
st.sidebar.markdown("**Required feature order:**")
for i, c in enumerate(columns, 1):
    st.sidebar.write(f"{i:02d}. {c}")

tab1, tab2 = st.tabs(["🔹 Single Prediction", "🔸 Batch CSV & Evaluation"])

# --------- Tab 1: Single Prediction ---------
with tab1:
    st.subheader("Single Prediction")
    st.caption("Paste a comma-separated line of 30 numbers (matching the required feature order).")

    default_line = ", ".join(["0"] * len(columns))
    user_line = st.text_area("Input values (comma-separated)", value=default_line, height=120)

    if st.button("Predict (single)"):
        try:
            values = [float(x.strip()) for x in user_line.split(",")]
            if len(values) != len(columns):
                st.error(f"Expected {len(columns)} values, got {len(values)}.")
            else:
                arr = np.array(values, dtype=float).reshape(1, -1)
                proba = float(model.predict_proba(arr)[0, 1])
                pred = int(proba >= thr)
                st.success(f"Prediction: **{pred}** (1=benign, 0=malignant)")
                st.metric("Probability of class 1 (benign)", f"{proba:.4f}")
                st.caption(f"Threshold = {thr:.2f} → Pred = 1 if prob ≥ threshold")
        except Exception as e:
            st.exception(e)

# --------- Tab 2: Batch CSV ---------
with tab2:
    st.subheader("Batch CSV Prediction & Optional Evaluation")
    st.caption("Upload a CSV with header containing all required columns. Optionally include a `target` column to compute metrics/plots.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
            st.write("**Preview:**", df.head())

            X = ensure_schema(df, columns)
            proba = model.predict_proba(X)[:, 1]
            pred = predict_with_threshold(proba, thr)

            out = df.copy()
            out["pred"] = pred
            out["prob"] = proba

            st.download_button(
                "⬇️ Download predictions",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
            st.success("Predictions generated.")

            # If user also provides ground-truth target, show evaluation
            if "target" in df.columns:
                y_true = df["target"].astype(int).values
                cm = confusion_matrix(y_true, pred)
                render_confusion_matrix(cm)

                # metrics
                try:
                    auc = roc_auc_score(y_true, proba)
                    st.metric("ROC AUC", f"{auc:.3f}")
                except Exception:
                    st.info("ROC AUC not available (check y values).")

                # curves
                try:
                    render_roc(y_true, proba)
                    render_pr(y_true, proba)
                except Exception:
                    st.info("Could not render curves — ensure `target` is binary 0/1.")

                # classification report (text)
                st.subheader("Classification Report")
                report_txt = classification_report(y_true, pred, digits=4)
                st.text(report_txt)

            else:
                st.info("Add a `target` column (0/1) in your CSV to view metrics and charts.")

        except Exception as e:
            st.exception(e)
