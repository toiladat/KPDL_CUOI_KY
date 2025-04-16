import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Đọc dữ liệu từ file CSV
df = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv")

# Lọc những dòng có label == 1
df_label1 = df[df['label'] == 1].copy()

# Hàm chuyển chuỗi vector thành numpy array
def parse_feature_vector(s):
    if isinstance(s, str):
        return np.array([float(num.strip()) for num in s.split(',')])
    return s

def parse_feature_vector2(s):
    if isinstance(s, str):
        s = s.replace('"', '')
        return np.array([float(num.strip()) for num in s.split(',')])
    return s

# Chuyển cột 'feature' thành vector
df_label1['features_parsed'] = df_label1['feature'].apply(parse_feature_vector)
X = np.stack(df_label1['features_parsed'].values)

# Áp dụng KMeans
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_label1['cluster'] = kmeans.fit_predict(X)

# Gán nhãn nguồn dữ liệu
df_label1['final_cluster'] = df_label1['cluster']
df_label1['source'] = 'OLD'

# Đọc file candidate mới
df_new = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\candidate_newspaper.csv")
df_new = df_new.head(5).copy()

# Chuyển vector feature thành mảng numpy
df_new['features_parsed'] = df_new['feature_vector'].apply(parse_feature_vector2)
X_new = np.stack(df_new['features_parsed'].values)

# Dự đoán cụm cho dữ liệu mới
df_new['predicted_cluster'] = kmeans.predict(X_new)
df_new['final_cluster'] = df_new['predicted_cluster']
df_new['source'] = 'NEW'

# Gộp dữ liệu cũ và mới
df_all = pd.concat([
    df_label1[['candidate', 'final_cluster', 'source']],
    df_new[['candidate', 'final_cluster', 'source']]
], ignore_index=True)

# In ra toàn bộ candidate theo từng cụm
for cluster_num in range(n_clusters):
    print(f"=== CLUSTER {cluster_num} ===")
    cluster_items = df_all[df_all['final_cluster'] == cluster_num]
    for _, row in cluster_items.iterrows():
        print(f"[{row['source']}] {row['candidate']}")
    print()