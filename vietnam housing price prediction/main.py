import pandas as pd

df = pd.read_csv('vietnam_housing_dataset.csv')

# print(df.head())
# print(df.describe())
# print(df.info())

# print(df.isnull().sum())

# print("Trước khi dropna:", df.shape)  # xem số hàng, cột ban đầu

# df = df.dropna(axis=1, thresh=len(df)*0.5)

# print("Sau khi dropna:", df.shape)     # xem số hàng, cột sau khi xóa

# cols_before = set(df.columns)  # lưu danh sách cột trước
df = df.dropna(axis=1, thresh=len(df)*0.5)
# cols_after = set(df.columns)   # lưu danh sách cột sau

# dropped_cols = cols_before - cols_after
# print("Các cột bị xóa:", dropped_cols)

# print("📊 Trước khi xử lý missing values:")
# print(df.head(100))   # in 100 dòng đầu

# Xử lý missing values
# 1️⃣ Điền median cho cột số
df = df.fillna(df.median(numeric_only=True))

# 2️⃣ Điền mode / "Unknown" cho cột chữ
for col in df.select_dtypes(include='object'):
    fill_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
    df[col] = df[col].fillna(fill_value)

# print("\n✅ Sau khi xử lý missing values:")
# print(df.head(100))   # in lại 100 dòng đầu

# print(df.isnull().sum())

# print(df.shape)

corr = df.corr(numeric_only=True)
# print(corr['Price'].sort_values(ascending=False))

# Chọn các cột có tương quan cao nhất với 'Price'
# (Bathrooms, Bedrooms, Floors là 3 feature mạnh nhất)
features = ['Bathrooms', 'Bedrooms', 'Floors']
target = 'Price'

# Giữ lại cột cần thiết
df = df[features + [target]]

# Kiểm tra nhanh
# print(df.head())
# print(df.corr()['Price'].sort_values(ascending=False))


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("trained successfully")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Tính các chỉ số đánh giá
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

