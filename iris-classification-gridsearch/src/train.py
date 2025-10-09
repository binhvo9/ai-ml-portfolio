import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import os, json, joblib



# load iris dataset
data = load_iris(as_frame=True)
X = data.data
y = data.target

print("Total NaN: ", X.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size: ", X_train.shape, y_train.shape)
print("Class distribution (train): ")
print(y_train.value_counts(normalize=True).round(2))

import numpy as np

# ==== 2) Pipeline + Baseline model ====
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC())
])

# ==== 3) GridSearch ====
param_grid = { #too see which combo best
    "clf__kernel": ["rbf", "linear"],
    "clf__C": [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto"]  # dùng cho rbf
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n=== Best Params ===")
print(grid.best_params_)
print("Best CV Acc:", round(grid.best_score_, 4))

# ==== 4) Đánh giá trên test set ====
y_pred = grid.predict(X_test)
print("\n=== Test Accuracy ===")
print(y_pred)
print(round(accuracy_score(y_test, y_pred), 4))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=data.target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("../artifacts", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# Lưu best model (pipeline đã GridSearch)
joblib.dump(grid.best_estimator_, "../artifacts/iris_svc_pipeline.joblib")

# Lưu metrics
metrics = {
    "best_params": grid.best_params_,
    "cv_accuracy": round(grid.best_score_, 4),
    "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}
with open("../results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved model to artifacts/iris_svc_pipeline.joblib")
print("Saved metrics to results/metrics.json")

