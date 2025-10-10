# Chúng ta sẽ xem dữ liệu nhanh và chia thành 2 phần: train và test.

import pandas as pd               # pandas: dụng cụ đọc và xử lý bảng dữ liệu (giống Excel)
from pathlib import Path          # Path: cách chỉ đường tới file/thư mục cho gọn
from sklearn.model_selection import train_test_split  # hàm chia dữ liệu ngẫu nhiên
import matplotlib.pyplot as plt   # vẽ hình đơn giản (cột, đường, ...)

INP = Path("data/processed/mini.parquet")  # đường dẫn file dữ liệu đã làm sạch
OUT_DIR = Path("data/processed")            # nơi sẽ lưu file train/test
ASSETS = Path("assets"); ASSETS.mkdir(parents=True, exist_ok=True)
# tạo thư mục "assets" nếu chưa có để lưu hình vẽ

df = pd.read_parquet(INP)  # đọc dữ liệu từ file mini.parquet vào bảng df

# ---------- Phần 1: EDA siêu nhanh (In vài thông tin ra màn hình) ----------
print("\n=== Shape:", df.shape)
# df.shape: cho biết bảng có bao nhiêu hàng và bao nhiêu cột

print("\n=== Numeric describe ===")
# .describe(): tóm tắt các cột số (số lượng, trung bình, nhỏ nhất, lớn nhất,…)
print(df[["price_nzd","accommodates","bedrooms","bathrooms",
          "minimum_nights","number_of_reviews","review_scores_rating"]].describe())

print("\n=== room_type top 5 ===")
# .value_counts(): đếm xem mỗi loại phòng xuất hiện bao nhiêu lần
print(df["room_type"].value_counts().head())

# ---------- Phần 2: Vẽ 1 biểu đồ nhỏ để nhìn phân phối giá ----------
plt.figure()                     # tạo tờ giấy vẽ mới
df["price_nzd"].hist(bins=60)    # vẽ biểu đồ cột (histogram) cho cột giá
plt.title("Nightly price (NZD)") # đặt tiêu đề hình
plt.xlabel("NZD")                # ghi tên trục ngang là “NZD”
plt.ylabel("Count")              # ghi tên trục dọc là “Count” (số lượng)
plt.tight_layout()               # sắp xếp cho hình gọn gàng, không bị cắt chữ
plt.savefig(ASSETS / "price_hist.png", dpi=120)
# lưu bức hình thành file "assets/price_hist.png"

# ---------- Phần 3: Chia dữ liệu thành Train/Test (80%/20%) ----------
train_df, test_df = train_test_split(
    df,                 # bảng dữ liệu cần chia
    test_size=0.2,      # phần test là 20% (còn lại 80% là train)
    random_state=42     # con số “ngẫu nhiên có kiểm soát”, để lần sau chia ra giống hệt
)

OUT_DIR.mkdir(parents=True, exist_ok=True)  # đảm bảo thư mục đích có tồn tại
train_df.to_parquet(OUT_DIR / "train.parquet", index=False)  # lưu phần train
test_df.to_parquet(OUT_DIR / "test.parquet", index=False)    # lưu phần test

# ---------- In thông báo hoàn thành ----------
print("\nSaved:", OUT_DIR / "train.parquet", len(train_df), "rows")
# báo đã lưu file train và cho biết có bao nhiêu dòng
print("Saved:", OUT_DIR / "test.parquet", len(test_df), "rows")
# báo đã lưu file test và cho biết có bao nhiêu dòng
print("Plot:", ASSETS / "price_hist.png")
# báo nơi đã lưu hình vẽ
