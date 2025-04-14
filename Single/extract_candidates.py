import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.corpus import stopwords

def extract_candidates(title, abstract):
    Kc = set()
    text = f"{title} {abstract}"
    text = re.sub(r'[^\w\s\-]', '', text)
    words = word_tokenize(text)
    tagged = pos_tag(words)

    # Ngữ pháp: cụm từ gồm tính từ (JJ) và danh từ (NN)
    grammar = "NP: {(<JJ.*>|<NN.*>)+<NN.*>}"
    cp = RegexpParser(grammar)
    tree = cp.parse(tagged)

    stop_words = set(stopwords.words('english'))

    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        phrase_tokens = [word for word, _ in subtree.leaves()]
        lower_tokens = [word.lower() for word in phrase_tokens if word.lower() not in stop_words]

        # Kiểm tra nếu không có từ nào là viết hoa toàn bộ, thì mới thêm cụm
        if not any(word.isupper() for word in phrase_tokens):
            if 2 <= len(lower_tokens) <= 4:
                phrase = ' '.join(lower_tokens)
                Kc.add(phrase)

    # Thêm các từ viết hoa toàn bộ (từ đơn chuyên ngành)
    for word, tag in tagged:
        if word.isupper() and word.lower() not in stop_words and len(word) > 1:
            Kc.add(word)  # Giữ nguyên viết hoa

    return sorted(Kc)

