def assign_label(candidate: str, text: str, frequency_threshold: int = 10, min_phrase_length: int = 4) -> int:
    """
    Gán nhãn cho candidate dựa trên tần suất xuất hiện và độ dài cụm từ.

    Args:
        candidate (str): Cụm từ cần gán nhãn.
        text (str): Văn bản chứa candidate.
        frequency_threshold (int): Ngưỡng tần suất để gán nhãn dương.
        min_phrase_length (int): Số lượng từ tối thiểu để gán nhãn dương.

    Returns:
        int: Nhãn 1 (từ khóa tiềm năng) hoặc 0 (không phải).
    """
    num_words = len(candidate.split())
    candidate_freq = text.lower().count(candidate.lower())
    return 1 if (candidate_freq >= frequency_threshold or num_words >= min_phrase_length) else 0
