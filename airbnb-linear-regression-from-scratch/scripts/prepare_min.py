import re, pandas as pd
# import re: dụng cụ để xử lý chữ/số kiểu pattern (ví dụ xóa dấu $ ,)
# import pandas as pd: thư viện để đọc và xử lý bảng dữ liệu

from pathlib import Path
# Path: cách gọn gàng để chỉ đường tới file/thư mục

RAW = Path("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/airbnb-linear-regression-from-scratch/data/raw/listings.csv")        # nơi đặt file gốc (CSV) vừa tải
OUT = Path("data/processed/mini.parquet")  # nơi sẽ lưu file đã làm sạch
OUT.parent.mkdir(parents=True, exist_ok=True)
# tạo thư mục 'data/processed' nếu chưa có, để lát nữa lưu file

def to_float_price(s):
    # hàm nhỏ để đổi giá tiền dạng chữ thành số (ví dụ "$1,234" → 1234.0)
    if pd.isna(s): return None
    # nếu ô trống (NaN) thì trả về None (không có gì)
    return float(re.sub(r"[^\d.]", "", str(s)))
    # re.sub(...): xóa mọi thứ KHÔNG phải số (0-9) hoặc dấu chấm
    # ví dụ "$1,234.00" -> "1234.00", rồi đổi sang kiểu số float

use_cols = ["id","price","room_type","neighbourhood_cleansed",
            "accommodates","bedrooms","bathrooms","minimum_nights",
            "number_of_reviews","review_scores_rating","latitude","longitude"]
# danh sách các cột mình muốn lấy từ file (chỉ lấy cái cần thiết cho gọn)

df = pd.read_csv(RAW, low_memory=False, usecols=lambda c: c in use_cols)
# đọc file CSV thành bảng 'df'
# low_memory=False: đọc chắc chắn, ít lỗi kiểu dữ liệu
# usecols=...: chỉ đọc các cột có tên nằm trong use_cols (nhanh + nhẹ)

df["price_nzd"] = df["price"].map(to_float_price)
# tạo cột mới 'price_nzd' là giá tiền đã đổi sang số (NZD)

df = df[(df["price_nzd"] >= 20) & (df["price_nzd"] <= 3000)]
# giữ lại các dòng có giá hợp lý (từ 20 đến 3000 NZD/đêm)
# giá quá nhỏ hoặc quá to thì bỏ (vì dễ là lỗi/ngoại lệ xấu)

df = df.dropna(subset=["price_nzd","room_type","accommodates","latitude","longitude"])
# xóa các dòng bị trống ở những cột quan trọng (vì thiếu thì khó học mô hình)

df.to_parquet(OUT, index=False)
# lưu bảng đã làm sạch thành file 'mini.parquet' (định dạng nhanh/gọn)

print("Done:", OUT, "rows:", len(df))
# in ra thông báo xong và có bao nhiêu dòng dữ liệu

print(df.head(3))
# in thử 3 dòng đầu cho mình nhìn xem ổn chưa
