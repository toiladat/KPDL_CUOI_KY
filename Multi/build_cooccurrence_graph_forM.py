import numpy as np
import itertools

def build_cooccurrence_graph_forM(all_documents_candidates):
    """
    Xây dựng đồ thị đồng xuất hiện từ danh sách các candidate keyword của nhiều bài báo.
    
    Parameters:
        all_documents_candidates (List[List[str]]): 
            Danh sách gồm nhiều danh sách con, mỗi danh sách con chứa candidate keywords của một bài báo.
    
    Returns:
        A (np.ndarray): Ma trận kề (adjacency matrix) kích thước n x n.
        vocab (List[str]): Danh sách tất cả các candidate keyword (ứng với các đỉnh trong đồ thị).
    """
    # Kiểm tra đầu vào: đảm bảo là danh sách các danh sách
    if not isinstance(all_documents_candidates, list):
        raise ValueError("Đầu vào phải là một danh sách.")
    for idx, candidates in enumerate(all_documents_candidates):
        if not isinstance(candidates, list):
            raise ValueError(f"Phần tử thứ {idx} của đầu vào phải là một danh sách.")

    # Tập hợp tất cả candidate keywords (loại bỏ trùng lặp) và gán chỉ số
    vocab_set = set(itertools.chain.from_iterable(all_documents_candidates))
    vocab = sorted(vocab_set)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    n = len(vocab)

    # Tạo ma trận kề rỗng
    A = np.zeros((n, n), dtype=int)

    # Với mỗi bài báo, tăng trọng số các cặp keyword đồng xuất hiện
    for candidates in all_documents_candidates:
        # Đảm bảo rằng mỗi candidate là chuỗi
        unique_candidates = {str(word) for word in candidates}
        indices = [vocab_index[word] for word in unique_candidates if word in vocab_index]
        for i, j in itertools.combinations(indices, 2):
            A[i][j] += 1
            A[j][i] += 1  # Vì là đồ thị vô hướng

    return A, vocab
