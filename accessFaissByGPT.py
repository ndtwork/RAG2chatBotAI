import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 📥 1. Tải FAISS index
index = faiss.read_index("vector_index.faiss")

# 📥 2. Tải metadata (để truy xuất đoạn văn gốc)
with open("chunk_metadata.pkl", "rb") as f:
    paragraphs = pickle.load(f)

# 🚀 3. Khởi tạo mô hình nhúng
embedding_model = SentenceTransformer("keepitreal/vietnamese-sbert")


def preprocess_text(text):
    """Chuẩn hóa văn bản trước khi nhúng"""
    return text.lower().strip()


def retrieve_relevant_chunks(question, index, paragraphs, embedding_model, top_k=3):
    """Tìm các đoạn văn bản liên quan từ FAISS"""
    # 🔹 1. Chuẩn hóa & mã hóa câu hỏi
    question_vector = embedding_model.encode([preprocess_text(question)])

    # 🔹 2. Chuẩn hóa vector để dùng Cosine Similarity
    faiss.normalize_L2(question_vector)

    # 🔹 3. Tìm top_k vector gần nhất trong FAISS
    distances, indices = index.search(question_vector, top_k)

    # 🔹 4. Lấy các đoạn văn bản tương ứng
    relevant_chunks = [paragraphs[idx] for idx in indices[0] if idx != -1]

    return relevant_chunks


# 🔍 Ví dụ câu hỏi
question = "Quy định về điểm thi thế nào?"

# 🔥 Gọi hàm truy xuất
relevant_chunks = retrieve_relevant_chunks(question, index, paragraphs, embedding_model)

# 📝 In kết quả
print("Các đoạn văn bản liên quan:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"{i}. {chunk}")
