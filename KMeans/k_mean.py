import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Đường dẫn file
TRAINING_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv"
PREDICTED_KEYWORD_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\predicted_keywords.csv"
OUTPUT_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\combined_cluster_info.csv"

def parse_feature_vector(s):
    """Chuyển chuỗi feature vector thành numpy array.
       Loại bỏ các dấu ngoặc kép, khoảng trắng thừa và chuyển các phần tử thành số thực.
    """
    if isinstance(s, str):
        s = s.replace('"', '').strip()
        return np.array([float(num.strip()) for num in s.split(',')])
    return s

def k_mean_keywords():
    # Bước 1: Đọc dữ liệu từ file training và file predicted_keyword
    try:
        df_training = pd.read_csv(TRAINING_FILE)
        df_keywords = pd.read_csv(PREDICTED_KEYWORD_FILE)
    except FileNotFoundError as e:
        print(f"⚠️ Lỗi: {e}")
        exit()
    
    # Lấy danh sách keyword cần phân cụm từ file predicted_keyword
    # (Giả sử cột chứa keyword có tên 'keyword')
    df_keywords = df_keywords[df_keywords['probability'] >= 0.5].head(100)

    keyword_list = df_keywords['keyword'].tolist()
    
    # Lọc những dòng trong training_bayes.csv tương ứng với các keyword đã được dự đoán
    df_keywords_training = df_training[df_training['candidate'].isin(keyword_list)].copy()
    df_keywords_training.drop_duplicates(subset='candidate', inplace=True)
    print (df_keywords_training)
    
    # Nếu muốn đảm bảo dữ liệu không bị null hay lỗi thì ép kiểu label nếu cần (tùy chọn)
    df_keywords_training['label'] = pd.to_numeric(df_keywords_training['label'], errors='coerce')
    
    # Bước 2: Chuyển cột 'feature' thành vector số (numpy array)
    df_keywords_training['features_parsed'] = df_keywords_training['feature'].apply(parse_feature_vector)
    df_keywords_training = df_keywords_training.dropna(subset=['features_parsed'])
    
    # Kiểm tra xem có dữ liệu để phân cụm không
    if df_keywords_training.empty:
        print("⚠️ Không có keyword nào trong dữ liệu training khớp với file predicted_keyword.")
        return

    # Bước 3: Tạo ma trận X từ cột feature đã chuyển đổi
    X = np.stack(df_keywords_training['features_parsed'].values)
    
    # Chọn số cụm (điều chỉnh số cụm nếu cần)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Phân cụm và lưu nhãn cụm vào cột mới 'cluster'
    df_keywords_training['cluster'] = kmeans.fit_predict(X)
    
    # Bước 4: Tổng hợp thông tin cụm
    cluster_data = []
    for cluster_num in range(n_clusters):
        members = df_keywords_training[df_keywords_training['cluster'] == cluster_num]['candidate'].tolist()
        centroid = kmeans.cluster_centers_[cluster_num]
        cluster_data.append({
            'cluster': cluster_num,
            'centroid': centroid.tolist(),  # Chuyển centroid thành list để lưu dưới dạng số
            'members': ', '.join(members),
            'num_members': len(members)
        })
    
    # Bước 5: Lưu kết quả ra file CSV
    pd.DataFrame(cluster_data).to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Đã lưu thông tin các cụm vào file: {OUTPUT_FILE}")

