# Mục tiêu: Huấn luyện nhiều model, chọn model tốt nhất theo RMSE, rồi lưu lại
# cả "pipeline" (gồm bước xử lý dữ liệu + model) để dùng dự đoán sau này.

from pathlib import Path                 # Path: chỉ đường tới file/thư mục
import time                              # time: đo thời gian
import joblib                            # joblib: lưu và mở model đã huấn luyện
import pandas as pd                      # pandas: đọc/ghi bảng dữ liệu
import numpy as np                       # numpy: làm việc với số
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Các model sẽ thử (giống bước lazy)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --------- Đường dẫn vào/ra ----------
DATA_DIR = Path("data/processed")          # nơi có train/test
TRAIN = DATA_DIR / "train.parquet"
TEST  = DATA_DIR / "test.parquet"
ART = Path("artifacts")                    # nơi lưu model sau khi train
ART.mkdir(parents=True, exist_ok=True)     # tạo thư mục nếu chưa có

# --------- 1) Đọc dữ liệu ----------
train_df = pd.read_parquet(TRAIN)          # đọc bảng train
test_df  = pd.read_parquet(TEST)           # đọc bảng test

# --------- 2) Chọn cột dùng để học ----------
TARGET = "price_nzd"                       # cột mục tiêu: giá NZD/đêm

NUM_COLS = [                               # cột số
    "accommodates", "bedrooms", "bathrooms",
    "minimum_nights", "number_of_reviews",
    "review_scores_rating", "latitude", "longitude",
]
CAT_COLS = ["room_type"]                   # cột chữ (loại phòng)

# Bỏ những dòng thiếu TARGET cho chắc
train_df = train_df.dropna(subset=[TARGET]).copy()
test_df  = test_df.dropna(subset=[TARGET]).copy()

# X (đầu vào) và y (đầu ra)
X_train = train_df[NUM_COLS + CAT_COLS]
y_train = train_df[TARGET].values
X_test  = test_df[NUM_COLS + CAT_COLS]
y_test  = test_df[TARGET].values

# --------- 3) Bộ xử lý đặc trưng tối giản ----------
# Với cột số: điền giá trị thiếu bằng median + chuẩn hoá
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Với cột chữ: điền "giá trị hay gặp nhất" + OneHot (chuyển chữ thành 0/1)
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

# Ghép 2 phần số + chữ lại
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ]
)

# --------- 4) Danh sách model để thử ----------
CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge(alpha=1.0)": Ridge(alpha=1.0, random_state=42),
    "Lasso(alpha=0.001)": Lasso(alpha=0.001, random_state=42, max_iter=10000),
    "ElasticNet(a=0.001,l1=0.3)": ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=42, max_iter=10000),
    "SGDRegressor": SGDRegressor(random_state=42, max_iter=2000),
    "KNN(k=5)": KNeighborsRegressor(n_neighbors=5),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest(100)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

# --------- 5) Thử từng model, tính điểm, chọn tốt nhất ----------
best_name = None                 # tên model tốt nhất (tạm thời chưa biết)
best_rmse = float("inf")         # RMSE tốt nhất (ban đầu để rất lớn)
best_pipe = None                 # pipeline tốt nhất (để lưu lại)
rows = []                        # chỗ ghi điểm từng model

for name, model in CANDIDATE_MODELS.items():         # duyệt qua từng model
    pipe = Pipeline(steps=[("prep", preprocessor),    # gắn preprocessor
                           ("model", model)])         # + model

    t0 = time.time()                                  # đo thời gian bắt đầu train
    pipe.fit(X_train, y_train)                        # học từ dữ liệu train
    train_secs = time.time() - t0                     # thời gian đã dùng

    y_pred = pipe.predict(X_test)                     # dự đoán trên test

    mae  = mean_absolute_error(y_test, y_pred)        # MAE: sai lệch tuyệt đối TB
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
    r2   = r2_score(y_test, y_pred)                   # R^2: càng gần 1 càng tốt

    rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "train_time_sec": train_secs})

    # Nếu RMSE nhỏ hơn cái tốt nhất hiện tại → cập nhật model tốt nhất
    if rmse < best_rmse:
        best_rmse = rmse
        best_name = name
        best_pipe = pipe

# --------- 6) In bảng kết quả và model tốt nhất ----------
results = pd.DataFrame(rows).sort_values(by="RMSE").reset_index(drop=True)
print("\n=== Model leaderboard (sorted by RMSE, lower is better) ===")
print(results.round({"MAE":3,"RMSE":3,"R2":4,"train_time_sec":3}).head(10))
print(f"\nBest model: {best_name}  |  RMSE={best_rmse:.3f}")

# --------- 7) Lưu pipeline tốt nhất ---------
out_path = ART / "best_pipeline.joblib"               # nơi lưu file model
joblib.dump(best_pipe, out_path)                      # lưu pipeline
print("\nSaved best pipeline to:", out_path)

# Thêm lưu bảng kết quả ra CSV cho dễ xem lại
(ART / "leaderboard.csv").write_text(results.to_csv(index=False))
print("Saved leaderboard to:", ART / "leaderboard.csv")
