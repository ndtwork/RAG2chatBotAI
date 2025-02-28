import os
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# ✅ Load mô hình DeepSeek-Embedding
model_name = "deepseek-ai/deepseek-embedding-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ✅ Đọc tệp văn bản
file_path = 'quyche.txt'  # Thay bằng tệp của bạn
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# ✅ Chia nhỏ văn bản thành đoạn (Chunking)
def split_into_chunks(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap  # Chồng lấn overlap từ
    return chunks

# Chia văn bản thành đoạn nhỏ
chunks = split_into_chunks(text)

# ✅ Hàm chuyển văn bản thành vector
def embed_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Lấy trung bình vector
    return embeddings.numpy()

# Chuyển đổi các đoạn văn bản thành vector
vectors = embed_text(chunks)

# ✅ Tạo FAISS index và lưu vector
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors))

# Lưu index vào file
faiss.write_index(index, "vector_index.faiss")

# Lưu danh sách chunks để truy xuất
with open("chunks.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("✅ Đã tạo và lưu vector thành công!")
