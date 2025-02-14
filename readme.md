## 1. Cài đặt thư viện cần thiết
Chạy lệnh sau để cài đặt thư viện:

    pip install langchain sentence-transformers faiss-cpu docx2txt chromadb pypdf

langchain: Hỗ trợ xử lý văn bản và kết nối với Ollama.
sentence-transformers: Tạo vector embeddings từ văn bản.
faiss-cpu: Dùng để lưu trữ và tìm kiếm vector.
docx2txt: Đọc nội dung từ file DOCX.
chromadb: Dùng để lưu embeddings (nếu không dùng FAISS).
pypdf: Đọc tài liệu PDF (nếu bạn có file PDF).

## 2. Đọc và Trích xuất Nội dung từ Tài Liệu

Nếu tài liệu của bạn là DOCX, dùng docx2txt để đọc nội dung:

```
import docx2txt

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

document_text = extract_text_from_docx("your_document.docx")
print(document_text[:500])  # Xem 500 ký tự đầu tiên

```
Nếu tài liệu là PDF, dùng pypdf:

```
from PyPDF2 import PdfReader

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

document_text = extract_text_from_pdf("your_document.pdf")
```
## 3. Chia Nhỏ Nội Dung Thành Các Đoạn Nhỏ (Chunking)
Mô hình không thể xử lý văn bản quá dài, nên ta cần chia nhỏ nội dung:

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

chunks = split_text(document_text)
print(f"Số đoạn văn bản sau khi chia: {len(chunks)}")
```

## 4. Tạo Vector Embeddings và Lưu Trữ

Dùng FAISS để lưu trữ các embeddings của tài liệu:
```
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Tạo embeddings từ mô hình Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Lưu vector embeddings vào FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("faiss_index")

```
## 5. Khi Người Dùng Hỏi, Tìm Kiếm Nội Dung Phù Hợp
Dùng FAISS để tìm nội dung liên quan:

```
# Tải lại FAISS để tìm kiếm

vectorstore = FAISS.load_local("faiss_index", embeddings)

query = "Quy định về điểm danh trong lớp học?"
retrieved_docs = vectorstore.similarity_search(query, k=3)

for doc in retrieved_docs:
    print(doc.page_content)
```
## 6. Tích Hợp với Ollama để Tạo Câu Trả Lời
Sau khi lấy được nội dung liên quan, ta dùng Ollama để sinh câu trả lời:
```
import ollama
query = "Quy định về điểm danh trong lớp học?"
retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

response = ollama.chat(
    model="deepseek-r1:1.5b",
    messages=[
        {"role": "system", "content": "Bạn là một trợ lý AI hỗ trợ sinh viên dựa trên tài liệu."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": retrieved_texts}
    ]
)
print(response["message"])

```
## 7. Tạo API với FastAPI để Kết Nối Web
Nếu bạn muốn chatbot chạy trên web, có thể tạo API bằng FastAPI:

    from fastapi import FastAPI
    import ollama
    
    app = FastAPI()
    
    @app.post("/chat")
    async def chat(query: str):
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])
    
        response = ollama.chat(
            model="deepseek-r1:1.5b",
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý AI hỗ trợ sinh viên."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": retrieved_texts}
            ]
        )
    
        return {"answer": response["message"]}`

Chạy API bằng lệnh:
 
uvicorn filename:app --reload
(Sửa filename thành tên file Python của bạn.)






addition link to ref :[RAG with OLLAMA](https://dev.to/mohsin_rashid_13537f11a91/rag-with-ollama-1049)