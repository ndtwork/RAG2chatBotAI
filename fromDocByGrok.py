# Cài đặt các thư viện cần thiết trước khi chạy code:
# pip install sentence-transformers faiss-cpu

import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Đường dẫn đến tệp văn bản đầu vào
file_path = 'quyche.txt'  # Thay bằng đường dẫn thực tế đến tệp của bạn

# 1. Đọc nội dung của tệp văn bản
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 2. Hàm chia văn bản thành các đoạn có độ dài tối đa 100 từ
# def split_into_chunks(text, max_words=100):
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     for word in words:
#         current_chunk.append(word)
#         if len(current_chunk) >= max_words:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = []
#     if current_chunk:  # Thêm chunk cuối cùng nếu còn từ
#         chunks.append(' '.join(current_chunk))
#     return chunks

def split_into_chunks(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        if end > len(words):
            end = len(words)
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap  # Di chuyển start về phía sau, chồng lấn overlap từ
    return chunks

# Chia văn bản thành các đoạn
paragraphs = split_into_chunks(text, max_words=100)

# Loại bỏ các đoạn trống
paragraphs = [para.strip() for para in paragraphs if para.strip()]

# 3. Khởi tạo mô hình Sentence Transformers
model = SentenceTransformer('keepitreal/vietnamese-sbert')  # Mô hình phù hợp cho tiếng Việt

# 4. Chuyển đổi các đoạn văn bản thành vector
vectors = model.encode(paragraphs)

# 5. In thông tin cơ bản về vector
print(f"Số lượng vector: {len(vectors)}")
print(f"Kích thước của mỗi vector: {vectors[0].shape}")

# 6. Lưu trữ vector bằng FAISS (tùy chọn)
# Chuyển đổi vectors thành numpy array
vectors = np.array(vectors)

# Tạo index FAISS sử dụng khoảng cách L2
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Lưu index vào tệp
faiss.write_index(index, 'vector_index.faiss')

print("Đã tạo và lưu index vector thành công!")