import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser
from itertools import combinations

# Tải các gói dữ liệu cần thiết (chạy lần đầu)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def build_cooccurrence_graph(text, window=3):
    """
    Xây dựng đồ thị đồng xuất hiện từ văn bản.
    
    Các bước thực hiện:
    1. Tách văn bản thành các câu.
    2. Với mỗi câu, token hóa và gán nhãn từ loại.
    3. Chỉ giữ lại các từ đơn (không chứa khoảng trắng) thuộc loại danh từ (NN*) hoặc tính từ (JJ*).
    4. Quét theo cửa sổ (window) với kích thước cho trước và tạo ra các cặp từ đồng xuất hiện.
    5. Đếm số lần xuất hiện đồng thời của mỗi cặp từ và lưu vào đồ thị dưới dạng dictionary lồng nhau.
    
    Parameters:
        text (str): Văn bản gốc.
        window (int): Kích thước cửa sổ để xác định các từ đồng xuất hiện.
        
    Returns:
        dict: Đồ thị đồng xuất hiện ở dạng dictionary, với mỗi từ là khóa và giá trị là dictionary chứa
              các từ kề cạnh cùng số lần đồng xuất hiện.
    """
    # Tách văn bản thành các câu
    sentences = sent_tokenize(text)
    
    tokens = []
    
    # Với mỗi câu: tokenize và gán nhãn POS, chỉ giữ các từ đơn là danh từ hoặc tính từ.
    for sent in sentences:
        token_list = word_tokenize(sent)
        pos_tags = pos_tag(token_list)
        # Lọc: chỉ lấy các từ mà khi dùng .split() có độ dài = 1 (đơn từ)
        filtered = [word.lower() for word, tag in pos_tags 
                    if (tag.startswith("NN") or tag.startswith("JJ")) and len(word.split()) == 1]
        tokens.extend(filtered)
    
    # Xây dựng đồ thị đồng xuất hiện (co-occurrence graph)
    graph = {}
    # Quét theo cửa sổ: từ vị trí i đến i+window - 1
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i:i+window]
        # Tạo các cặp từ từ cửa sổ (sử dụng combinations)
        for w1, w2 in combinations(window_tokens, 2):
            if w1 == w2:
                continue
            # Cập nhật mối liên hệ từ w1 sang w2
            if w1 not in graph:
                graph[w1] = {}
            graph[w1][w2] = graph[w1].get(w2, 0) + 1
            
            # Và cập nhật ngược lại từ w2 sang w1
            if w2 not in graph:
                graph[w2] = {}
            graph[w2][w1] = graph[w2].get(w1, 0) + 1
            
    return graph

