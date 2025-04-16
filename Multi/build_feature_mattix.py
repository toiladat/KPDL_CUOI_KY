import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_feature_matrix(doc_candidates, vocab):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    n = len(vocab)
    p = len(doc_candidates)
    X = np.zeros((n, p + 1))

    for doc_id, candidates in enumerate(doc_candidates):
        for pos, word in enumerate(candidates):
            if word in vocab_index:
                idx = vocab_index[word]
                X[idx, 0] += 1  # frequency
                X[idx, doc_id + 1] = (pos + 1) / len(candidates)  # normalized position

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
