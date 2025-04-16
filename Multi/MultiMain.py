
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from extract_candidates import extract_candidates
from Multi.build_cooccurrence_graph_forM import build_cooccurrence_graph_forM
from sklearn.preprocessing import MinMaxScaler
from build_feature_mattix import create_feature_matrix
from build_label_mattrix import create_label_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------- GCN Model ------------
class MGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super(MGCN, self).__init__()
        self.gcn1 = nn.Linear(in_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # binary classification (keyword / not keyword)
        )

    def forward(self, A_hat, X):
        H1 = F.relu(torch.matmul(A_hat, self.gcn1(X)))
        H2 = F.relu(torch.matmul(A_hat, self.gcn2(H1)))
        out = self.mlp(H2)
        return out

# ---------- Preprocess ----------
def normalize_adjacency(A):
    A_tilde = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A_tilde, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat


# ---------- Main Pipeline ----------
def MultiMain():
    df = pd.read_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\data_format_title_abstract.csv")
    all_candidates = []
    for _, row in df.iterrows():
        title = str(row['title'])
        abstract = str(row['abstract'])
        cands = extract_candidates(title, abstract)
        all_candidates.append(cands)

    print("✅ Trích xuất từ khóa xong.")


    A, vocab = build_cooccurrence_graph_forM(all_candidates)
    A_hat = normalize_adjacency(A)
    print("✅ Chuẩn hóa ma trận kề.")

    X = create_feature_matrix(all_candidates, vocab)
    L = create_label_matrix(all_candidates, vocab)

    A_hat = torch.FloatTensor(A_hat)
    X = torch.FloatTensor(X)
    L = torch.LongTensor(L)

    model = MGCN(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    mask = (L != -1)  # Không có từ nào bị loại bỏ
    epochs = 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(A_hat, X)
        loss = criterion(logits[mask], L[mask])
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred[mask] == L[mask]).float().mean()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")

    print("🎉 Training hoàn tất!")

    # Dự đoán
    model.eval()
    with torch.no_grad():
        y_pred = model(A_hat, X)
        probs = F.softmax(y_pred, dim=1)[:, 1]  # Xác suất là từ khóa thật

    # Lưu kết quả
    sorted_indices = np.argsort(-probs.numpy())
    result_df = pd.DataFrame({
        'keyword': [vocab[i] for i in sorted_indices],
        'probability': probs.numpy()[sorted_indices],
        'label': L.numpy()[sorted_indices]
    })
    result_df.to_csv(r"E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\predicted_keywords.csv", index=False)
    print("📄 Đã lưu kết quả vào predicted_keywords.csv")
