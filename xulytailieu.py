import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ ĐÚNG
from langchain_community.vectorstores import FAISS

# 🟢 1. Trích xuất nội dung từ PDF
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:  # Kiểm tra nếu có văn bản
                text += extracted_text + "\n"
    return text

document_text = extract_text_from_pdf("QUYCHE_HUST2023.pdf")
print(document_text[:500])  # Xem trước 500 ký tự đầu tiên

# 🟢 2. Chia nhỏ văn bản thành các đoạn nhỏ
def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

chunks = split_text(document_text)  # 🟢 Không bị ghi đè!

print(f"Số đoạn văn bản sau khi chia: {len(chunks)}")

# 🟢 3. Tạo vector embeddings từ văn bản đã xử lý
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(chunks, embeddings)  # 🟢 Không dùng danh sách mẫu nữa!

# 🟢 4. Truy vấn dữ liệu
query = "Quy định về thi cử là gì?"
retrieved_docs = vectorstore.similarity_search(query, k=3)

# 🟢 5. Hiển thị kết quả tìm kiếm
for doc in retrieved_docs:
    print(doc.page_content)
