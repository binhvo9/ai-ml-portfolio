# Mục tiêu: Mở "pipeline tốt nhất" đã lưu → nhận vài thông tin nhà → đoán giá (NZD/đêm)

from pathlib import Path                 # Path: chỉ đường tới file/thư mục
import argparse                          # argparse: đọc tham số từ dòng lệnh
import joblib                            # joblib: mở file model đã lưu
import pandas as pd                      # pandas: tạo bảng 1 hàng làm input

# ----- 1) Nơi đặt file model đã lưu -----
ART = Path("artifacts")                  # thư mục chứa model
MODEL_PATH = ART / "/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/airbnb-linear-regression-from-scratch/scripts/artifacts/best_pipeline.joblib"  # đường dẫn tới file model

# ----- 2) Tạo "bảng 1 hàng" từ tham số người dùng -----
def make_input_df(args):
    # Tạo một bảng (DataFrame) có 1 dòng, đúng tên các cột mà pipeline mong muốn
    data = {
        "accommodates": [args.accommodates],        # số người ở
        "bedrooms": [args.bedrooms],                # số phòng ngủ
        "bathrooms": [args.bathrooms],              # số phòng tắm
        "minimum_nights": [args.minimum_nights],    # số đêm tối thiểu
        "number_of_reviews": [args.number_of_reviews],  # tổng review
        "review_scores_rating": [args.review_scores_rating],  # điểm đánh giá (0-100)
        "latitude": [args.latitude],                # vĩ độ (tọa độ)
        "longitude": [args.longitude],              # kinh độ (tọa độ)
        "room_type": [args.room_type],              # loại phòng (ví dụ: 'Entire home/apt')
    }
    return pd.DataFrame(data)

# ----- 3) Hàm chính: mở model → dự đoán -----
def main():
    # a) Khai báo các tham số cần nhập khi chạy file
    parser = argparse.ArgumentParser(
        description="Predict nightly Airbnb price (NZD) with saved pipeline"
    )
    # Thêm từng tham số (đều bắt buộc, giúp model hiểu căn nhà của bạn)
    parser.add_argument("--accommodates", type=int, required=True, help="Số khách tối đa (ví dụ 4)")
    parser.add_argument("--bedrooms", type=float, required=True, help="Số phòng ngủ (ví dụ 2)")
    parser.add_argument("--bathrooms", type=float, required=True, help="Số phòng tắm (ví dụ 1)")
    parser.add_argument("--minimum_nights", type=int, required=True, help="Số đêm tối thiểu (ví dụ 2)")
    parser.add_argument("--number_of_reviews", type=int, required=True, help="Tổng số review (ví dụ 30)")
    parser.add_argument("--review_scores_rating", type=float, required=True, help="Điểm rating 0-100 (ví dụ 92)")
    parser.add_argument("--latitude", type=float, required=True, help="Vĩ độ Auckland (ví dụ -36.87)")
    parser.add_argument("--longitude", type=float, required=True, help="Kinh độ Auckland (ví dụ 174.73)")
    parser.add_argument("--room_type", type=str, required=True, help="Loại phòng (Entire home/apt | Private room | Shared room | Hotel)")

    args = parser.parse_args()           # đọc các tham số người dùng đã nhập

    # b) Kiểm tra file model có tồn tại không
    if not MODEL_PATH.exists():
        print("❌ Không tìm thấy model. Hãy chạy: python scripts/train_and_save_min.py")
        return

    # c) Mở pipeline đã lưu (bên trong có cả bước xử lý dữ liệu + model)
    pipe = joblib.load(MODEL_PATH)

    # d) Tạo bảng 1 hàng từ tham số (đúng tên cột mà pipeline cần)
    X_new = make_input_df(args)

    # e) Dùng pipeline để dự đoán giá (NZD/đêm)
    y_pred = pipe.predict(X_new)[0]

    # f) In kết quả đẹp mắt
    print("\n=== Prediction ===")
    print(f"Ước tính giá/đêm (NZD): {y_pred:,.2f}")
    print("\nGợi ý: thử đổi room_type hoặc số khách để xem giá thay đổi ra sao.")

# ----- 4) Chạy hàm chính khi file được gọi trực tiếp -----
if __name__ == "__main__":
    main()
