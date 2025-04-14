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

# Đọc dữ liệu
df = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data_format_title_abstract.csv")

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
df_train.to_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv", index=False)
print("✅ Training data saved to training_bayes.csv (candidate, feature, label)")

# -----------------------------------------------------------------------------------------
# Process new paper for testing
new_paper = {
    "title": "Newspaper: Neural Networks Applications in Healthcare",
    "abstract": ("Neural networks are extensively used in deep learning for healthcare applications. "
                 "cost model for outlier detection procedures . In addition , we provide experimental results from the application of our algorithms on a Minneapolis-St . Paul -LRB- Twin Cities -RRB- traffic dataset to show their effectiveness and usefulness . B Introduction Data mining is a process to extract nontrivial , previously unknown and potentially useful infor - mation -LRB- such as knowledge rules , constraints , regularities -RRB- from data in databases -LSB- 11 , 4 -RSB- . The explosive growth in data and databases used in business management , government administra - tion , and scientific data analysis has created a need for tools that can automatically transform the processed data into useful information and knowledge . Spatial data mining is a process of discovering interesting and useful but implicit spatial patterns . With the enormous amounts of spatial data obtained from satellite images , medical images , GIS , etc. , it is a nontrivial task for humans to explore spatial data in detail . Spatial data sets and patterns are abundant in many application domains related to NASA , the National Imagery and Mapping Agency -LRB- NIMA -RRB- , the National Cancer Institute -LRB- NCI -RRB- , and the Unite States Department of Transportation -LRB- USDOT -RRB- . Data Mining tasks can be classified into four general categories : -LRB- a -RRB- dependency detection -LRB- e.g. , association rules -RRB- -LRB- b -RRB- class identification -LRB- e.g. , classification , clustering -RRB- -LRB- c -RRB- class description -LRB- e.g. , concept generalization -RRB- , and -LRB- d -RRB- exception/outlier detection -LSB- 9 -RSB- . The objective of the first three categories is to identify patterns or rules from a significant portion of a data set . On the other hand , the outlier detection problem focuses on the identification of a very small subset of data objects often viewed as noises , errors , exceptions , or deviations . Outliers have been informally defined as observations which appear to be inconsistent with the remainder of that set of data -LSB- 2 -RSB- , or which deviate so much from other observations so as to arouse suspicions that they were generated by a different mechanism -LSB- 6 -RSB- . The identification of outliers can lead to the discovery of unexpected knowledge and has a number of practical applications in areas such as credit card fraud , the performance analysis of athletes , voting irregularities , bankruptcy , and weather prediction . Outliers in a spatial data set can be classified into three categories : set-based outliers , multi-dimensional space-based outliers , and graph-based outliers . A set-based outlier is a data object whose attributes are inconsistent ")
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
df_test_results.to_csv("E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\candidate_newspaper.csv", index=False)
print("✅ Test results saved to candidate_newspaper.csv (candidate, predicted_probability)")

# In ra top 5 từ khóa có xác suất cao nhất
print("\nTop 5 Keywords:")
for kw, score in results:
    print(f"{kw}: {score:.3f}")
