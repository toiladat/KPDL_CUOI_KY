# read_data.py
import pandas as pd

def read_and_process_data():
    filepath = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data_format_title_abstract.csv"

    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(filepath)

    # Xử lý giá trị thiếu trong cột title và abstract
    df[["title", "abstract"]] = df[["title", "abstract"]].fillna("")

    # Chuyển thành danh sách các bài báo (dictionary)
    papers = df[["title", "abstract"]].to_dict(orient="records")

    # Tạo văn bản tổng hợp từ tất cả các bài báo
    all_text = " ".join([p["title"] + " " + p["abstract"] for p in papers])

    return papers, all_text
