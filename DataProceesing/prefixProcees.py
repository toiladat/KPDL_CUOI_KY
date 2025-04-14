import pandas as pd
import ast

# Đọc dữ liệu từ file gốc
df = pd.read_csv("E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data.csv")  # thay bằng đường dẫn thật

# Tạo danh sách lưu kết quả
processed_rows = []

for _, row in df.iterrows():
    try:
        id_val = row["id"]
        doc_list = ast.literal_eval(row["document"])  # chuyển chuỗi -> list

        if "T" in doc_list and "A" in doc_list:
            t_index = doc_list.index("T")
            a_index = doc_list.index("A")

            title_words = doc_list[t_index + 1:a_index]
            abstract_words = doc_list[a_index + 1:]

            # Ghép lại thành chuỗi văn bản
            title = " ".join(w for w in title_words if w.strip() not in ["", "--", '""'])
            abstract = " ".join(w for w in abstract_words if w.strip() not in ["", "--", '""'])

            processed_rows.append({
                "id": id_val,
                "title": title,
                "abstract": abstract
            })

    except Exception as e:
        print(f"Lỗi ở dòng id={row['id']}: {e}")
        continue

# Tạo DataFrame mới
processed_df = pd.DataFrame(processed_rows)

# Lưu ra file CSV mới
processed_df.to_csv("data_processed.csv", index=False)

print("✅ Đã lưu thành công vào data_processed.csv")
