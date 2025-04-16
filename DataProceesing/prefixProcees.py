import pandas as pd
import ast

def extract_title_abstract():
    input_path = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data_raw.csv"
    output_path = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data_format_title_abstract.csv"

    df = pd.read_csv(input_path)
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

                title = " ".join(w for w in title_words if w.strip() not in ["", "--", '""'])
                abstract = " ".join(w for w in abstract_words if w.strip() not in ["", "--", '""'])

                processed_rows.append({
                    "id": id_val,
                    "title": title,
                    "abstract": abstract
                })

        except Exception as e:
            print(f"Lỗi ở dòng id={row.get('id', 'N/A')}: {e}")
            continue

    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(output_path, index=False)
    print(f"✅ Tiền xủ lý thành công")
