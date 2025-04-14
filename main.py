import random
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from sklearn.naive_bayes import GaussianNB
from itertools import combinations

# Import NLTK và tải một số resource cần thiết
import nltk
from nltk import word_tokenize, pos_tag, RegexpParser, sent_tokenize

# nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

import re
from nltk import word_tokenize, pos_tag
from nltk.chunk import RegexpParser
##################################
# 1. TRÍCH XUẤT CANDIDATE KEYWORDS VỚI NLTK
##################################
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def extract_candidates(title, abstract):
    # Bước 1: Khởi tạo danh sách từ khóa
    Kc = []

    # Bước 2: Xử lý loại bỏ dấu câu
    title_clean = remove_punctuation(title)
    abstract_clean = remove_punctuation(abstract)

    # Bước 3: Tách từ (sửa preserve_line=True để tránh lỗi punkt_tab)
    text_clean = title_clean + " " + abstract_clean
    tokens = word_tokenize(text_clean, preserve_line=True)

    # Bước 4: Gắn nhãn từ loại (POS tagging)
    tagged = pos_tag(tokens)

    # Bước 5: Định nghĩa ngữ pháp trích xuất cụm danh từ & tính từ
    grammar = r"KT:{(<JJ>*<NN.*>)?<JJ>*<NN.*> +}"  # danh từ với tính từ phía trước
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(tagged)

    # Bước 6–10: Lấy cụm phù hợp
    for subtree in tree.subtrees():
        if subtree.label() == 'KT':
            phrase = " ".join(word for word, pos in subtree.leaves())
            word_count = len(subtree.leaves())

            if word_count >= 2 and phrase not in Kc:
                Kc.append(phrase)
            elif word_count == 1:
                word = subtree.leaves()[0][0]
                if word.isupper() and word not in Kc:
                    Kc.append(word)

    return Kc

##################################
# 2. XÂY DỰNG ĐỒ THỊ ĐỒNG XUẤT VỚI NLTK
##################################
def build_cooccurrence_graph(text, window=3):
    """
    Xây dựng đồ thị đồng xuất hiện từ văn bản.
    Ở đây, ta tách câu và sau đó tokenize, POS tag bằng NLTK để chỉ lấy ra các token với POS là danh từ (NN*) hoặc tính từ (JJ*).
    Output: Dictionary {word: {neighbor: weight, ...}, ...}
    """
    sentences = sent_tokenize(text)
    tokens = []
    for sent in sentences:
        token_list = word_tokenize(sent)
        pos_tags = pos_tag(token_list)
        # Lọc token theo POS: chỉ lấy danh từ (NN, NNP, NNPS, ...) hoặc tính từ (JJ, JJR, JJS)
        filtered = [word.lower() for word, tag in pos_tags if tag.startswith("NN") or tag.startswith("JJ")]
        tokens.extend(filtered)
    graph = {}
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i:i + window]
        for w1, w2 in combinations(window_tokens, 2):
            if w1 == w2:
                continue
            if w1 in graph and w2 in graph[w1]:
                graph[w1][w2] += 1
            else:
                if w1 not in graph:
                    graph[w1] = {}
                graph[w1][w2] = 1
            # Cập nhật đối xứng
            if w2 in graph and w1 in graph[w2]:
                graph[w2][w1] += 1
            else:
                if w2 not in graph:
                    graph[w2] = {}
                graph[w2][w1] = 1
    return graph

##################################
# 3. RANDOM WALK VÀ HUẤN LUYỆN WORD2VEC
##################################
def perform_random_walks(graph, num_walks=10, walk_length=10):
    """
    Thực hiện random walks trên đồ thị đồng xuất hiện.
    Output: List các chuỗi (mỗi chuỗi là list các từ).
    """
    walks = []
    nodes = list(graph.keys())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                curr = walk[-1]
                neighbors = list(graph[curr].keys())
                if neighbors:
                    weights = [graph[curr][nbr] for nbr in neighbors]
                    next_word = random.choices(neighbors, weights=weights)[0]
                    walk.append(next_word)
                else:
                    break
            walks.append(walk)
    return walks

##################################
# 4. LẤY VECTOR EMBEDDING CHO CANDIDATE
##################################
def get_candidate_embedding(candidate, w2v_model):
    """
    Tách candidate thành các từ, lấy vector của từng từ từ mô hình Word2Vec,
    sau đó tính trung bình để có vector biểu diễn cho candidate.
    """
    words = candidate.lower().split()
    vectors = []
    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

##################################
# 5. TÍNH CÁC FEATURE BỔ SUNG CHO CANDIDATE
##################################
def compute_additional_features(candidate, text):
    """
    Tính 3 đặc trưng:
      - f1: Tần suất xuất hiện (số lần candidate xuất hiện / tổng số từ)
      - f2: Vị trí xuất hiện đầu tiên (index của candidate / tổng số từ)
      - f3: Độ dài candidate (số ký tự / 20)
    """
    words = text.lower().split()
    total = len(words) if len(words) > 0 else 1
    f1 = words.count(candidate) / total
    try:
        pos = words.index(candidate)
        f2 = pos / total
    except ValueError:
        f2 = 1.0
    f3 = len(candidate) / 20.0
    return np.array([f1, f2, f3])

##################################
# 6. TẠO DỮ LIỆU HUẤN LUYỆN TỪ 20 BÀI BÁO
##################################
papers = []
for i in range(20):
    if i % 2 == 0:
        title = f"Paper {i}: Deep Learning Advances"
        abstract = ("Deep neural networks have revolutionized modern AI. "
                    "Advancements in deep learning and neural computation drive research.")
        true_keywords = {"deep", "neural", "deep neural networks", "deep learning"}
    else:
        title = f"Paper {i}: Machine Learning Trends"
        abstract = ("Machine learning techniques and data mining are trending. "
                    "Innovations in machine algorithms and data analysis are crucial.")
        true_keywords = {"machine", "data", "machine learning", "data mining"}
    papers.append({"title": title, "abstract": abstract, "true_keywords": true_keywords})

# Tổng hợp văn bản của tất cả các bài báo để huấn luyện mô hình embedding (Word2Vec)
all_text = ""
for paper in papers:
    all_text += paper["title"] + " " + paper["abstract"] + " "

# Xây dựng đồ thị đồng xuất hiện cho toàn bộ tập văn bản dựa trên các token là danh từ và tính từ
graph = build_cooccurrence_graph(all_text, window=3)
walks = perform_random_walks(graph, num_walks=10, walk_length=10)
w2v_model = Word2Vec(sentences=walks, vector_size=50, window=3, min_count=1, workers=1)

# Tạo tập dữ liệu huấn luyện: Với mỗi candidate của mỗi bài báo, tính vector đặc trưng và gán nhãn (thủ công)
X_train = []    # Feature vector cho candidate
y_train = []    # Nhãn: 1 nếu candidate nằm trong true_keywords, 0 nếu không
all_candidates = []

for paper in papers:
    title = paper["title"]
    abstract = paper["abstract"]
    true_keywords = set([kw.lower() for kw in paper["true_keywords"]])
    text = title + " " + abstract
    candidates = extract_candidates(title, abstract)  # candidate từ các noun phrase và từ đơn (NN, JJ)
    for cand in candidates:
        emb = get_candidate_embedding(cand, w2v_model)
        additional = compute_additional_features(cand, text)
        # Ghép đặc trưng bổ sung (3 chiều) với embedding (50 chiều) thành vector 53 chiều
        feature_vector = np.concatenate([additional, emb])
        label = 1 if cand in true_keywords else 0
        X_train.append(feature_vector)
        y_train.append(label)
        all_candidates.append(cand)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Tổng số candidate trên tập train:", len(all_candidates))
print("Ví dụ 10 candidate đầu:")
for cand, lab in zip(all_candidates[:10], y_train[:10]):
    print(cand, "->", lab)

##################################
# 7. HUẤN LUYỆN MÔ HÌNH Gaussian Naive Bayes
##################################
model = GaussianNB()
model.fit(X_train, y_train)
print("\nMô hình GaussianNB đã được huấn luyện.")

##################################
# 8. DỰ ĐOÁN CHO 1 BÀI BÁO MỚI (newspaper article)
##################################
new_paper = {
    "title": "Newspaper: Neural Networks Applications in Healthcare",
    "abstract": ("Neural networks are extensively used in deep learning for healthcare applications. "
                 "Recent advances in neural computation have significantly improved diagnostic accuracy.")
}

new_text = new_paper["title"] + " " + new_paper["abstract"]
new_candidates = extract_candidates(new_paper["title"], new_paper["abstract"])

X_test = []
for cand in new_candidates:
    emb = get_candidate_embedding(cand, w2v_model)
    additional = compute_additional_features(cand, new_text)
    feature_vector = np.concatenate([additional, emb])
    X_test.append(feature_vector)
X_test = np.array(X_test)

# Dự đoán xác suất với GaussianNB, trả về xác suất cho label 1 (candidate là từ khóa chính)
probas = model.predict_proba(X_test)[:, 1]

results = []
for cand, prob in zip(new_candidates, probas):
    results.append((cand, prob))

results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

print("\nKết quả dự đoán cho bài báo mới (candidate và xác suất là từ khóa):")
for cand, prob in results_sorted:
    print(f"Candidate: {cand}, Xác suất: {prob:.3f}")

top_5 = results_sorted[:5]
print("\nTop 5 candidate được chọn làm từ khóa chính:")
for cand, prob in top_5:
    print(f"{cand}: {prob:.3f}")
