import numpy as np
from gensim.models import Word2Vec

def get_candidate_embedding(candidate, w2v_model):
    words = candidate.lower().split()
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)