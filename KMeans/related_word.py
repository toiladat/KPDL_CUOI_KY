import pandas as pd
import numpy as np
from ast import literal_eval

# Đường dẫn file
CLUSTER_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\combined_cluster_info.csv"
NEW_CANDIDATE_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\candidate_newspaper.csv"
TRAINING_FILE = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\training_bayes.csv"

def parse_feature_vector(s):
    """Chuyển chuỗi feature vector thành numpy array"""
    if isinstance(s, str):
        s = s.replace('"', '').strip()
        return np.array([float(x.strip()) for x in s.split(',')])
    return s

def related_word():
    # Đọc dữ liệu
    try:
        df_clusters = pd.read_csv(CLUSTER_FILE, converters={
            'centroid': literal_eval,
            'members': lambda x: x.split(', ')
        })
        df_training = pd.read_csv(TRAINING_FILE)
        df_training['feature'] = df_training['feature'].apply(parse_feature_vector)
    except FileNotFoundError as e:
        print(f"⚠️ Lỗi: {e}")
        return

    df_new = pd.read_csv(NEW_CANDIDATE_FILE).head(5)
    df_new['feature_vector'] = df_new['feature_vector'].apply(parse_feature_vector)
    df_new = df_new.dropna(subset=['feature_vector'])

    results = []
    for idx, row in df_new.iterrows():
        candidate_name = row['candidate']
        features = row['feature_vector']

        # Tính khoảng cách đến centroid
        centroid_distances = [np.linalg.norm(features - np.array(c)) for c in df_clusters['centroid']]
        closest_cluster_idx = np.argmin(centroid_distances)
        cluster_info = df_clusters.iloc[closest_cluster_idx]

        cluster_members = df_training[df_training['candidate'].isin(cluster_info['members'])]
        member_distances = []
        for _, member_row in cluster_members.iterrows():
            distance = np.linalg.norm(features - member_row['feature'])
            member_distances.append((member_row['candidate'], distance))

        member_distances = sorted(member_distances, key=lambda x: x[1])
        
        # Lấy top 3 từ gần nhất
        top3 = []
        for candidate, distance in member_distances:
            if candidate not in [t[0] for t in top3]:
                top3.append((candidate, distance))
            if len(top3) == 3:
                break

        results.append({
            'new_candidate': candidate_name,
            'assigned_cluster': cluster_info['cluster'],
            'top3_nearest': top3
        })

    # Lưu kết quả
    output_df = pd.DataFrame([{
        'new_candidate': r['new_candidate'],
        'cluster': r['assigned_cluster'],
        'nearest_1': r['top3_nearest'][0][0] if len(r['top3_nearest']) > 0 else None,
        'distance_1': r['top3_nearest'][0][1] if len(r['top3_nearest']) > 0 else None,
        'nearest_2': r['top3_nearest'][1][0] if len(r['top3_nearest']) > 1 else None,
        'distance_2': r['top3_nearest'][1][1] if len(r['top3_nearest']) > 1 else None,
        'nearest_3': r['top3_nearest'][2][0] if len(r['top3_nearest']) > 2 else None,
        'distance_3': r['top3_nearest'][2][1] if len(r['top3_nearest']) > 2 else None
    } for r in results])

    output_path = r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\nearest_candidates.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\n✅ Đã lưu kết quả tìm kiếm từ liên quan vào file nearest_candidates.csv")

