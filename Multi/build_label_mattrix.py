import numpy as np

def create_label_matrix(doc_candidates, vocab, threshold=3):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    count = np.zeros(len(vocab))

    for candidates in doc_candidates:
        for word in set(candidates):  # chỉ tính 1 lần mỗi bài
            if word in vocab_index:
                count[vocab_index[word]] += 1

    labels = (count >= threshold).astype(int)
    return labels
