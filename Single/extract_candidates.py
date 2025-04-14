import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser
from nltk.corpus import stopwords
import pandas as pd
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

def extract_candidates(title, abstract):
    stop_words = set(stopwords.words('english'))
    
    title = str(title) if pd.notna(title) else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    text = title + " " + abstract

    # Loại bỏ ký tự đặc biệt
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    sentences = sent_tokenize(text)
    candidates = set()

    # Grammar: Cụm danh từ (danh từ có thể có tính từ phía trước)
    grammar = "NP: {<JJ>*<NN.*>+}"
    cp = RegexpParser(grammar)
    
    for sent in sentences:
        tokens = word_tokenize(sent)
        pos_tags = pos_tag(tokens)

        # Cụm danh từ (nếu có nhiều hơn 1 từ)
        tree = cp.parse(pos_tags)
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            phrase = " ".join(word for word, tag in subtree.leaves())
            if (len(phrase.split()) > 1 and 
                all(w.lower() not in stop_words for w in phrase.split()) and
                phrase.replace(" ", "").isalpha()):
                candidates.add(phrase)

        # Từ đơn: chỉ giữ nếu viết HOA toàn bộ (ví dụ: CNN, USA)
        for word, tag in pos_tags:
            if word.isupper() and len(word) > 1:
                candidates.add(word)

    return list(candidates)
