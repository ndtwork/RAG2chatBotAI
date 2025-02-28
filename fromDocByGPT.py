import os
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Đọc file văn bản đầu vào
file_path = 'quyche.txt'  # Thay bằng đường dẫn file thực tế
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    return text.strip()

# Làm sạch văn bản
cleaned_text = clean_text(text)

# Chia nhỏ văn bản thành các đoạn với chồng lấn
def split_into_chunks(text, chunk_size=100, overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", ". ", " "]
    )
    return splitter.split_text(text)

# Chia văn bản thành các đoạn
paragraphs = split_into_chunks(cleaned_text, chunk_size=100)

# Loại bỏ các đoạn trống
paragraphs = [para.strip() for para in paragraphs if para.strip()]

# Khởi tạo mô hình Sentence Transformers
model = SentenceTransformer('keepitreal/vietnamese-sbert')  # Mô hình nhúng tiếng Việt

# Chuyển đổi các đoạn văn bản thành vector
vectors = model.encode(paragraphs)
vectors = np.array(vectors)

# Lưu metadata để giữ thứ tự các đoạn
chunk_metadata = [{"id": i, "text": para} for i, para in enumerate(paragraphs)]

# Lưu metadata vào file
with open("chunk_metadata.pkl", "wb") as f:
    pickle.dump(chunk_metadata, f)

# Tạo index FAISS sử dụng khoảng cách L2
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Lưu index vào file
faiss.write_index(index, 'vector_index.faiss')

print("✅ Đã tạo và lưu index vector thành công!")
