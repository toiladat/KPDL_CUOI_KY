import os
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
import joblib

# Thêm thư mục hiện tại (Single) và thư mục cha (CUOI_KY)
sys.path.append(os.path.dirname(__file__))  # Để import file cùng cấp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Để import Ultils từ cha

from Ultils.read_data import read_and_process_data

from extract_candidates import extract_candidates
from Single.build_cooccurrence_graph_forS import build_cooccurrence_graph_forS
from perform_random_walks import perform_random_walks
from get_candidate_embedding import get_candidate_embedding
from compute_additional_features import compute_additional_features
from assign_label import assign_label




# ========== HUẤN LUYỆN VÀ LƯU MÔ HÌNH ========== #
def train_model():
    papers, all_text = read_and_process_data()

    graph = build_cooccurrence_graph_forS(all_text)
    walks = perform_random_walks(graph)
    w2v_model = Word2Vec(sentences=walks, vector_size=3, window=3, min_count=1, workers=1)

    X_train, y_train = [], []
    candidates_all, features_all = [], []

    for paper in papers:
        text = paper["title"] + " " + paper["abstract"]
        candidates = extract_candidates(paper["title"], paper["abstract"])

        for cand in candidates:
            emb = get_candidate_embedding(cand, w2v_model)
            additional = compute_additional_features(cand, text)
            feature_vector = np.concatenate([additional, emb])
            label = assign_label(cand, text)

            X_train.append(feature_vector)
            y_train.append(label)
            candidates_all.append(cand)
            features_all.append(feature_vector)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = GaussianNB()
    model.fit(X_train, y_train)
    df_train = pd.DataFrame({
        "candidate": candidates_all,
        "feature": [",".join(map(str, feat)) for feat in features_all],  # Chuyển feature vector thành chuỗi
        "label": y_train
    })
    df_train.to_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv", index=False)
    print("✅ Training data saved to training_bayes.csv (candidate, feature, label)")

    # Lưu mô hình
    joblib.dump(model,  r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Model\model_naivebayes.pkl")
    print(f"✅ Mô hình đã được lưu tại: model_naivebayes.pkl")





# ========== HÀM TRÍCH XUẤT TỪ KHÓA ==========
def get_candidates_from_text(title, abstract):
    # Load model từ file
    if os.path.exists( r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Model\model_naivebayes.pkl"):
        print(f"📦 Đã tải mô hình từ NavieBayes")
        model = joblib.load( r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Model\model_naivebayes.pkl")
    else:
        raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại. Vui lòng huấn luyện trước.")

    # Xử lý văn bản
    text = title + " " + abstract
    graph = build_cooccurrence_graph_forS(text)
    walks = perform_random_walks(graph)
    w2v_model = Word2Vec(sentences=walks, vector_size=3, window=3, min_count=1, workers=1)

    # Trích xuất candidates và tính đặc trưng
    candidates = extract_candidates(title, abstract)
    X_test = []
    feature_map = {}
    for cand in candidates:
        emb = get_candidate_embedding(cand, w2v_model)
        additional = compute_additional_features(cand, text)
        feature_vector = np.concatenate([additional, emb])
        X_test.append(feature_vector)
        feature_map[cand] = feature_vector
    # Dự đoán xác suất
    X_test = np.array(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Sắp xếp kết quả theo xác suất giảm dần
    results = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)

    # Lưu kết quả vào DataFrame
    df_test_results = pd.DataFrame([{
        "candidate": cand,
        "feature_vector": '"' + ",".join(map(str, feature_map[cand])) + '"',
        "predicted_probability": prob
    } for cand, prob in results])

    df_test_results.to_csv(
        r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\candidate_newspaper.csv",
        index=False
    )

    print("✅ Đã lưu kết quả candidate_newspaper.csv với feature vector và xác suất dự đoán.")





