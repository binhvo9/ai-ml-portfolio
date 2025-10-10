# Mục tiêu: Chuẩn bị dữ liệu đơn giản → chạy nhiều model nhanh → so sánh điểm
# "Lazy predict" ở đây nghĩa là: ta tự viết hàm chạy lần lượt nhiều model,
# đo điểm số và thời gian, rồi in bảng kết quả.

from pathlib import Path                   # Path: chỉ đường tới file/thư mục
import time                                # time: đo thời gian chạy
import pandas as pd                        # pandas: đọc bảng dữ liệu
import numpy as np                         # numpy: làm việc với mảng số

# Các "dụng cụ" xử lý dữ liệu và model từ scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer        # điền giá trị thiếu
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer   # gắn các bước xử lý theo cột
from sklearn.pipeline import Pipeline           # nối các bước lại thành 1 đường ống
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Một vài model đơn giản, phổ biến
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --------- Đường dẫn file vào/ra ----------
DATA_DIR = Path("data/processed")                 # nơi có file train/test
TRAIN = DATA_DIR / "train.parquet"                # file train
TEST  = DATA_DIR / "test.parquet"                 # file test
ASSETS = Path("assets"); ASSETS.mkdir(parents=True, exist_ok=True)  # nơi lưu kết quả

# --------- 1) Đọc dữ liệu ----------
# Đọc 2 bảng đã chia sẵn (ở bước trước)
train_df = pd.read_parquet(TRAIN)
test_df  = pd.read_parquet(TEST)

# --------- 2) Chọn cột đơn giản ----------
# y (mục tiêu dự đoán) là giá tiền (số tiền NZD/đêm)
TARGET = "price_nzd"

# Các cột số (dùng trực tiếp)
NUM_COLS = [
    "accommodates", "bedrooms", "bathrooms",
    "minimum_nights", "number_of_reviews",
    "review_scores_rating", "latitude", "longitude",
]

# Các cột chữ (category) → ta sẽ biến thành số bằng One-Hot
CAT_COLS = ["room_type"]  # giữ đơn giản: chỉ lấy loại phòng

# Giữ lại các cột có sẵn, bỏ dòng thiếu TARGET
train_df = train_df.dropna(subset=[TARGET]).copy()
test_df  = test_df.dropna(subset=[TARGET]).copy()

# Lấy X (đầu vào) và y (đầu ra) cho train/test
X_train = train_df[NUM_COLS + CAT_COLS]
y_train = train_df[TARGET].values
X_test  = test_df[NUM_COLS + CAT_COLS]
y_test  = test_df[TARGET].values

# --------- 3) Tạo "bộ xử lý đặc trưng" tối giản ----------
# Với cột số: điền thiếu bằng median, rồi chuẩn hoá (cho các cột về cùng thang đo)
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # điền giá trị còn trống
    ("scaler", StandardScaler()),                    # chuẩn hoá
])

# Với cột chữ: điền thiếu bằng "missing", rồi One-Hot (biến chữ thành 0/1)
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

# Kết hợp 2 đường ống: số + chữ
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ]
)

# --------- 4) Danh sách model để "lazy predict" ----------
# Ta sẽ thử nhiều model khác nhau. Mỗi cái có tên dễ hiểu.
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

# --------- 5) Hàm "lazy_predict" tự code ----------
# Ý tưởng: lặp qua từng model → gắn chung với preprocessor thành Pipeline → train → đo điểm
def lazy_predict(models: dict, X_tr, y_tr, X_te, y_te):
    rows = []                               # nơi để gom kết quả
    for name, model in models.items():      # duyệt từng model theo tên
        # Ghép preprocessor + model thành một đường ống
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model),
        ])
        # Đo thời gian huấn luyện
        t0 = time.time()
        pipe.fit(X_tr, y_tr)                # học từ dữ liệu train
        train_secs = time.time() - t0

        # Dự đoán trên test
        y_pred = pipe.predict(X_te)

        # Tính điểm số:
        mae  = mean_absolute_error(y_te, y_pred)                     # MAE: sai lệch tuyệt đối trung bình
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))       # RMSE: căn bậc 2 của MSE
        r2   = r2_score(y_te, y_pred)                                # R^2: giải thích được bao nhiêu %

        # Lưu kết quả vào bảng
        rows.append({
            "model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "train_time_sec": round(train_secs, 3),
        })

    # Trả về DataFrame đã sắp xếp theo RMSE (nhỏ hơn là tốt hơn)
    res = pd.DataFrame(rows).sort_values(by="RMSE").reset_index(drop=True)
    return res

# --------- 6) Chạy "lazy predict" ----------
results = lazy_predict(CANDIDATE_MODELS, X_train, y_train, X_test, y_test)

# In vài dòng đầu tiên để nhìn nhanh
print("\n=== Lazy Predict Results (top) ===")
print(results.head(10))

# Lưu kết quả ra CSV để bạn mở xem cho dễ
out_csv = ASSETS / "lazy_results.csv"
results.to_csv(out_csv, index=False)
print("\nSaved results to:", out_csv)
