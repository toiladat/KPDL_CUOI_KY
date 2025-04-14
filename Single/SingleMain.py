import numpy as np
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from extract_candidates import extract_candidates
from build_cooccurrence_graph import build_cooccurrence_graph
from perform_random_walks import perform_random_walks
from get_candidate_embedding import get_candidate_embedding
from compute_additional_features import compute_additional_features
import pandas as pd
import nltk
from nltk import word_tokenize

# Đảm bảo các tài nguyên của NLTK đã được tải
nltk.download('punkt')

# Đọc dữ liệu
df = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\data_processed.csv")

# Xử lý NaN
df[["title", "abstract"]] = df[["title", "abstract"]].fillna("")

# Chuyển thành danh sách các bài báo
papers = df[["title", "abstract"]].to_dict(orient="records")

# Tạo văn bản tổng hợp từ toàn bộ tập
all_text = " ".join([p["title"] + " " + p["abstract"] for p in papers])

# Tạo graph từ co-occurrence
graph = build_cooccurrence_graph(all_text)

# Sinh random walks từ graph
walks = perform_random_walks(graph)

# Train Word2Vec
w2v_model = Word2Vec(sentences=walks, vector_size=3, window=3, min_count=1, workers=1)

# Ngưỡng dùng để gán nhãn
frequency_threshold = 10
min_phrase_length = 4

# Tạo danh sách lưu dữ liệu huấn luyện
X_train = []  # Feature vector
y_train = []  # Labels (0 or 1)
candidates_all = []  # Lưu candidates để lưu vào file CSV
features_all = []  # Lưu feature vectors để lưu vào file CSV

for paper in papers:
    text = paper["title"] + " " + paper["abstract"]
    candidates = extract_candidates(paper["title"], paper["abstract"])

    for cand in candidates:
        # Lấy embedding của candidate và tính các đặc trưng khác
        emb = get_candidate_embedding(cand, w2v_model)
        additional = compute_additional_features(cand, text)
        feature_vector = np.concatenate([additional, emb])

        # Tính các đặc trưng để gán nhãn
        num_words = len(cand.split())
        candidate_freq = text.lower().count(cand.lower())
        label = 1 if (candidate_freq >= frequency_threshold or num_words >= min_phrase_length) else 0

        # Lưu candidate, feature và nhãn vào danh sách
        candidates_all.append(cand)
        features_all.append(feature_vector)
        X_train.append(feature_vector)
        y_train.append(label)

# Chuyển X_train và y_train thành numpy array để huấn luyện mô hình
X_train = np.array(X_train)
y_train = np.array(y_train)

# Huấn luyện classifier với Naive Bayes
model = GaussianNB()

# Huấn luyện mô hình với cả label == 0 và label == 1
model.fit(X_train, y_train)

# Lưu dữ liệu huấn luyện vào CSV
df_train = pd.DataFrame({
    "candidate": candidates_all,
    "feature": [",".join(map(str, feat)) for feat in features_all],  # Chuyển feature vector thành chuỗi
    "label": y_train
})
df_train.to_csv("training_data.csv", index=False)
print("✅ Training data saved to training_data.csv (candidate, feature, label)")

# -----------------------------------------------------------------------------------------
# Process new paper for testing
new_paper = {
    "title": "Newspaper: Neural Networks Applications in Healthcare",
    "abstract": ("Neural networks are extensively used in deep learning for healthcare applications. "
                 "Recent advances in neural computation have significantly improved diagnostic accuracy.")
}

new_text = new_paper["title"] + " " + new_paper["abstract"]

# Trích xuất các candidate từ bài báo
candidates = extract_candidates(new_paper["title"], new_paper["abstract"])

# Tính toán các đặc trưng cho các candidates
X_test = []
for cand in candidates:
    # Lấy embedding của candidate từ mô hình word2vec đã huấn luyện
    emb = get_candidate_embedding(cand, w2v_model)
    # Tính các đặc trưng bổ sung
    additional = compute_additional_features(cand, new_text)
    # Ghép nối các đặc trưng bổ sung và embedding lại với nhau
    X_test.append(np.concatenate([additional, emb]))

# Dự đoán xác suất cho các candidates
X_test = np.array(X_test)
probas = model.predict_proba(X_test)[:, 1]  # Lấy xác suất của nhãn = 1

# Sắp xếp các candidates theo xác suất giảm dần
results = sorted(zip(candidates, probas), key=lambda x: x[1], reverse=True)

# Lưu kết quả dự đoán vào CSV
df_test_results = pd.DataFrame({
    "candidate": [kw for kw, _ in results],
    "predicted_probability": [score for _, score in results]
})
df_test_results.to_csv("test_results.csv", index=False)
print("✅ Test results saved to test_results.csv (candidate, predicted_probability)")

# In ra top 5 từ khóa có xác suất cao nhất
print("\nTop 5 Keywords:")
for kw, score in results[:5]:
    print(f"{kw}: {score:.3f}")
