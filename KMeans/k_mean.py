import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def k_mean():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv")

    # Ép kiểu label về int nếu cần
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # Lọc những dòng có label == 1
    df_label1 = df[df['label'] == 1].copy()

    # Hàm chuyển chuỗi vector thành numpy array
    def parse_feature_vector(s):
        if isinstance(s, str):
            s = s.replace('"', '').strip()
            return np.array([float(num.strip()) for num in s.split(',')])
        return s

    # Chuyển cột 'feature' thành vector
    df_label1['features_parsed'] = df_label1['feature'].apply(parse_feature_vector)
    df_label1 = df_label1.dropna(subset=['features_parsed'])

    # Tạo ma trận X
    X = np.stack(df_label1['features_parsed'].values)

    # Áp dụng KMeans
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_label1['cluster'] = kmeans.fit_predict(X)

    # Tạo dữ liệu cụm
    cluster_data = []
    for cluster_num in range(n_clusters):
        members = df_label1[df_label1['cluster'] == cluster_num]['candidate'].tolist()
        centroid = kmeans.cluster_centers_[cluster_num]
        cluster_data.append({
            'cluster': cluster_num,
            'centroid': centroid.tolist(),
            'members': ', '.join(members),
            'num_members': len(members)
        })

    # Lưu vào file
    output_path = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\combined_cluster_info.csv"
    pd.DataFrame(cluster_data).to_csv(output_path, index=False)

    print(f"✅ Đã lưu tất cả thông tin các cụm vào file combined_cluster_info.csv")

k_mean()
