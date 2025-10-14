import pandera as pa                     # 🧱 Gọi thư viện pandera để kiểm tra dữ liệu (data validator)
from pandera import Column, DataFrameSchema, Check  # 📦 Lấy 3 công cụ chính: cột, khung dữ liệu, và điều kiện kiểm tra
import pandas as pd                      # 🐼 Dùng pandas để làm việc với bảng dữ liệu

PriceSchema = DataFrameSchema(           # 🧩 Tạo "bản nội quy" cho bảng dữ liệu (schema)
    {
        "ts": Column(pa.DateTime, Check(lambda s: s.is_monotonic_increasing), nullable=False),
        # ⏰ Cột "ts" phải là kiểu ngày giờ, tăng dần (không đi lùi thời gian), không được để trống

        "price": Column(float, Check.gt(0.0), nullable=False),
        # 💰 Cột "price" là số thực, phải lớn hơn 0, không được để trống

        "market_cap": Column(float, Check.ge(0.0), nullable=True),
        # 🏦 Cột "market_cap" là số thực, >= 0, có thể để trống

        "volume": Column(float, Check.ge(0.0), nullable=True),
        # 📊 Cột "volume" là số thực, >= 0, có thể để trống

        "symbol": Column(str, nullable=False),
        # 🪙 Cột "symbol" là chữ (vd: BTC, ETH), không được để trống

        "freq": Column(str, nullable=False),
        # ⏱️ Cột "freq" là chữ (vd: '1h', '1d'), không được để trống
    },
    strict=True,     # 🚧 Không được thêm cột lạ ngoài danh sách này
    coerce=True,     # 🔄 Tự đổi kiểu dữ liệu cho đúng nếu có thể (vd: '10' → 10.0)
)
