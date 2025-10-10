import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1️⃣ đọc file
df = pd.read_csv("data/breast_cancer.csv")
print("Data shape:", df.shape)
print(df.head())

# 2️⃣ thống kê mô tả nhanh
print("\nSummary:")
print(df.describe())

# 3️⃣ kiểm tra target balance
print("\nTarget counts:\n", df["target"].value_counts())
df["target"].value_counts().plot(kind="bar", title="Target Distribution")
plt.show()

# 4️⃣ chia train/test (giữ tỉ lệ target)
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# 5️⃣ lưu
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("✅ Saved split data in /data")
