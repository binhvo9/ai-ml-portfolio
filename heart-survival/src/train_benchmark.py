import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1️⃣ load data
df = pd.read_csv("data/breast_cancer.csv")

X = df.drop("target", axis=1)
y = df["target"]

# 2️⃣ chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ chia data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 4️⃣ chạy benchmark
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# 5️⃣ show top model
print("=== Lazy Predict Results (top) ===")
print(models.head(10))
models.to_csv("artifacts/lazy_results.csv")
