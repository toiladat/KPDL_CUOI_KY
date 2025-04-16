import numpy as np

def compute_additional_features(candidate, text):
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


[
    [21,1,0,1,0],
    [10,1,0,0,1]

    
]