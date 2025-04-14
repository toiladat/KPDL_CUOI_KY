import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser
from nltk.corpus import stopwords
import pandas as pd
import re



def extract_candidates(title, abstract):
    stop_words = set(stopwords.words('english'))
    unwanted_abbreviations = {'nsf', 'nih', 'usa', 'uk'}
    unwanted_phrases = {'nsf grant', 'nih funding'}

    title = str(title) if pd.notna(title) else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    text = title + " " + abstract

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    sentences = sent_tokenize(text)
    candidates = set()

    grammar = "NP: {<JJ>?<NN.*><NN.*>?}"

    cp = RegexpParser(grammar)

    for sent in sentences:
        tokens = word_tokenize(sent)
        pos_tags = pos_tag(tokens)

        tree = cp.parse(pos_tags)
        for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            phrase = " ".join(word for word, tag in subtree.leaves())
            words = phrase.split()
            if (len(words) > 1 and
                all(w.lower() not in stop_words for w in words) and
                not any(w.lower() in unwanted_abbreviations for w in words) and
                phrase.replace(" ", "").isalpha()):
                candidates.add(phrase)

        for word, tag in pos_tags:
            if word.isupper() and len(word) > 1:
                candidates.add(word)

    candidates = {phrase for phrase in candidates if phrase.lower() not in unwanted_phrases}

    return list(candidates)