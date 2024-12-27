import pandas as pd
from sklearn.utils import resample

# Đọc dữ liệu vào DataFrame
df = pd.read_csv('data.csv')  # Thay 'your_data.csv' bằng đường dẫn file CSV của bạn

# Giả sử cột 'label' chứa nhãn và cột 'content' chứa dữ liệu văn bản
label_column = 'label'
content_column = 'content'

# Tách các nhóm nhãn
pos_samples = df[df[label_column] == 'POS']
neg_samples = df[df[label_column] == 'NEG']
neu_samples = df[df[label_column] == 'NEU']

# Giảm mẫu của nhãn POS xuống 50,000
pos_samples_reduced = resample(
    pos_samples,
    replace=False,  # Không lặp lại mẫu
    n_samples=7000,  # Số lượng mẫu mong muốn
    random_state=42   # Đảm bảo kết quả lặp lại được
)

# Kết hợp lại với các nhãn khác
df_balanced = pd.concat([pos_samples_reduced, neg_samples, neu_samples])

# Shuffle dữ liệu để đảm bảo ngẫu nhiên
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Xuất dữ liệu ra file CSV mới
df_balanced.to_csv('balanced_data.csv', index=False)
print("Dữ liệu đã được cân bằng và lưu vào 'balanced_data.csv'.")
