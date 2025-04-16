import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

def extract_candidates(title, abstract):
    """
    Trích xuất candidate keywords từ tiêu đề và tóm tắt theo các tiêu chí:
      1. Nếu candidate là 1 từ, nó phải chỉ chứa chữ cái và dấu gạch ngang, 
         được viết HOÀN TOÀN bằng chữ (uppercase) và có độ dài > 1.
      2. Nếu candidate là cụm 2 từ, nó phải có định dạng "ADJ N" tức:
            - Từ thứ nhất phải là tính từ (POS tag bắt đầu bởi 'JJ')
            - Từ thứ hai phải là danh từ (POS tag bắt đầu bởi 'NN')
         Đồng thời, mỗi từ phải có độ dài > 2 ký tự và không được viết hoàn toàn bằng chữ in hoa.
      3. Trước đó, chia văn bản thành các câu sau đó làm sạch từng câu bằng cách loại bỏ tất cả các ký tự không phải chữ, không phải khoảng trắng, nhưng giữ lại dấu gạch ngang.
    """
    # Kết hợp tiêu đề và tóm tắt
    text = f"{title} {abstract}"
    
    # Chia văn bản thành các câu
    sentences = sent_tokenize(text)
    
    # Làm sạch từng câu: loại bỏ tất cả ký tự không phải chữ, khoảng trắng và dấu gạch ngang
    cleaned_sentences = [re.sub(r'[^A-Za-z\s-]', '', sent) for sent in sentences]
    

    candidates = set()
    
    # Hàm kiểm tra từ hợp lệ: chỉ chứa chữ và dấu gạch ngang
    def is_valid_word(word):
        return bool(re.fullmatch(r'[A-Za-z-]+', word))
    
    # Duyệt qua từng câu đã làm sạch
    for sent in cleaned_sentences:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        
        # Tiêu chí 1: Candidate là từ đơn
        for word, tag in tagged:
            # Chỉ lấy từ nếu từ chỉ chứa chữ và dấu gạch ngang, được viết toàn hoa và có độ dài > 1.
            if is_valid_word(word) and word == word.upper() and len(word) > 1:
                candidates.add(word)
        
        # Tiêu chí 2: Candidate là cụm 2 từ theo định dạng "ADJ N" hoặc "N N"
        for i in range(len(tagged) - 1):
            word1, tag1 = tagged[i]
            word2, tag2 = tagged[i+1]
            # Kiểm tra cả hai từ hợp lệ
            if is_valid_word(word1) and is_valid_word(word2):
                # Mỗi từ phải có độ dài > 2 ký tự
                if len(word1) > 2 and len(word2) > 2:
                    # Kiểm tra định dạng "ADJ N" hoặc "N N"
                    # Và bổ sung: không được có từ nào viết hoàn toàn bằng chữ in hoa
                    if (
                        (tag1.startswith("JJ") and tag2.startswith("NN")) or
                        (tag1.startswith("NN") and tag2.startswith("NN"))
                    ) and not (word1.isupper() or word2.isupper()):
                        phrase = f"{word1} {word2}"
                        candidates.add(phrase)
    
    return sorted(candidates)