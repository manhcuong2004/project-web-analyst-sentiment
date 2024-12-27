import re
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from gensim.models import KeyedVectors
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO
from underthesea import word_tokenize

# Load mô hình CNN đã huấn luyện
model_sentiment = load_model("models.h5")
print("Sentiment model loaded successfully!")

# Load mô hình nhúng từ (Word2Vec)
file_path = 'baomoi.model.bin'
try:
    model_embedding = KeyedVectors.load_word2vec_format(file_path, binary=True)
    print("Embedding model loaded successfully!")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    sys.exit(1)

word_labels = list(model_embedding.key_to_index.keys())
max_seq = 100  # Chiều dài câu tối đa
embedding_size = 400

# Hàm chuyển câu thành ma trận nhúng từ
def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    for i in range(min(len(words), max_seq)):
        word = words[i]
        if word in word_labels:
            matrix[i] = model_embedding[word]
    return matrix

# Hàm loại bỏ emoji
def remove_emoji(text):
    for emoji in UNICODE_EMOJI.values():
        text = text.replace(emoji, "")
    for emoticon in EMOTICONS_EMO.values():
        text = text.replace(emoticon, "")
    return text

def clean_text(text):
    text = text.lower()
    text = remove_emoji(text) 
    text = re.sub(r'\d+', ' ', text)  
    text = re.sub(r'[^\w\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    
    words = word_tokenize(text, format="text")
    return words

# Tiền xử lý dữ liệu
def preprocess_data(text):
    text = clean_text(text)  # Làm sạch văn bản
    return text

# Dự đoán bằng mô hình Keras
def predict_with_cnn(text):
    text_cleaned = preprocess_data(text)
    matrix_embedding = np.expand_dims(comment_embedding(text_cleaned), axis=0)
    matrix_embedding = np.expand_dims(matrix_embedding, axis=3)
    result = model_sentiment.predict(matrix_embedding)
    return np.argmax(result)

# Hàm xử lý file CSV
def process_csv(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    # Kiểm tra và xử lý
    if 'content' in df.columns:
        labels = []  # Danh sách để lưu nhãn tương ứng
        for content in df['content']:
            cnn_prediction = predict_with_cnn(content)

            # Gán nhãn trực tiếp vào prediction
            if cnn_prediction == 0:
                labels.append("negative")
            elif cnn_prediction == 1:
                labels.append("neutral")
            elif cnn_prediction == 2:
                labels.append("positive")

        df['prediction'] = labels  # Ghi nhãn vào cột 'prediction'
        df.to_csv(output_csv_path, index=False)
        print(f"Kết quả đã được lưu vào file: {output_csv_path}")
    else:
        print("Cột 'content' không tồn tại trong file CSV.")
        print("Các cột có sẵn trong file CSV:", list(df.columns))
        sys.exit(1)



if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    process_csv(input_csv, output_csv)
