from pathlib import Path                 # 📂 Dùng để làm việc với đường dẫn file (path) dễ hơn

def save_csv(df, path: str):             # 💾 Hàm để lưu dữ liệu df (bảng) vào file ở chỗ path
    p = Path(path)                       # 🚪 Biến path thành kiểu "đường dẫn thông minh"
    p.parent.mkdir(parents=True, exist_ok=True)  # 🏗️ Nếu thư mục chưa có thì tạo luôn (khỏi lỗi)
    df.to_csv(p, index=False)            # 📜 Lưu bảng df thành file CSV, không kèm số thứ tự
