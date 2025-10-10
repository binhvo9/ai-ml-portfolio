# src/eval_viz.py
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    classification_report
)

ART = Path("artifacts")
DATA = Path("data")

MODEL_PATH = ART / "model.joblib"
COLUMNS_PATH = ART / "columns.json"
METRICS_PATH = ART / "metrics.eval.json"   # file mới cho bước eval

def load_artifacts_and_data():
    model = joblib.load(MODEL_PATH)
    columns = json.load(open(COLUMNS_PATH))
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")
    # Ensure order/float dtype
    X_test = X_test[columns].astype(float)
    return model, columns, X_test, y_test

def plot_confusion(cm, labels=("Malignant(0)", "Benign(1)"), out=ART / "cm.png"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    # write numbers
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, proba, out=ART / "roc.png"):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_pr(y_true, proba, out=ART / "pr.png"):
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main():
    ART.mkdir(parents=True, exist_ok=True)
    model, columns, X_test, y_test = load_artifacts_and_data()

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save plots
    plot_confusion(cm)
    plot_roc(y_test, proba)
    plot_pr(y_test, proba)

    # Save metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    json.dump(metrics, open(METRICS_PATH, "w"), indent=2)

    print("✅ Saved:")
    print(" - artifacts/cm.png")
    print(" - artifacts/roc.png")
    print(" - artifacts/pr.png")
    print(" - artifacts/metrics.eval.json")

if __name__ == "__main__":
    main()
