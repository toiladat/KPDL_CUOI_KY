from flask import Flask, render_template, request
import os
import sys
import pandas as pd  # Thêm dòng này để đọc CSV

# Add paths để import được các module như bạn đã làm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Single')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Multi')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Kmeans')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'DataProcessing')))

from Single.SingleMain import get_candidates_from_text
from KMeans.related_word import related_word

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        title = request.form['title']
        abstract = request.form['abstract']

        # Gọi hàm xử lý
        keywords = get_candidates_from_text(title, abstract)
        related_word()

        # Đọc CSV để lấy cột new_candidate
        df = pd.read_csv(r'E:\MON_TREN_LOP\KHAI_PHA_DU_LIEU\CUOI_KY\Data\nearest_candidates.csv')
        new_candidates = df['new_candidate'].dropna().tolist()

        result = {
            "keywords": keywords,
            "new_candidates": new_candidates
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
