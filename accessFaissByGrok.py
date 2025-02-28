import faiss

# Tải file FAISS
index = faiss.read_index('vector_index.faiss')

# Tải danh sách đoạn văn bản từ file
with open('quyche.txt', 'r', encoding='utf-8') as f:
    paragraphs = [line.strip() for line in f.readlines()]


from sentence_transformers import SentenceTransformer

# Khởi tạo mô hình nhúng
embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')


def retrieve_relevant_chunks(question, index, paragraphs, embedding_model, top_k=3):
    # Mã hóa câu hỏi thành vector
    question_vector = embedding_model.encode([question])

    # Tìm top_k vector gần nhất trong FAISS
    distances, indices = index.search(question_vector, top_k)

    # Lấy các đoạn văn bản tương ứng
    relevant_chunks = [paragraphs[idx] for idx in indices[0]]

    return relevant_chunks

# Ví dụ câu hỏi
question = "Quy định về điểm thi thế nào?"

# Gọi hàm truy xuất
relevant_chunks = retrieve_relevant_chunks(question, index, paragraphs, embedding_model)

# In kết quả
print("Các đoạn văn bản liên quan:")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"{i}. {chunk}")