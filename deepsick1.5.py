import requests
import pdfplumber
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Bước 1: Trích xuất văn bản từ PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Bước 2: Làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# Bước 3: Chia văn bản thành các đoạn nhỏ (chunking)
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

# Bước 4: Tạo embedding từ văn bản
def create_embeddings(text):
    embeddings = HuggingFaceEmbeddings(model_name="hieuduong29", api_key="YOUR_API_KEY")
    return embeddings.embed(text)

# Bước 5: Lưu trữ vector vào FAISS
def store_in_faiss(documents, embeddings):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("pdf_data_index")
    return vector_store

# Bước 6: Truy vấn tìm kiếm và trả lời từ chatbot
def query_ollama_model(query, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/v1/complete"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response.json() if response.status_code == 200 else None

# Main function để xử lý và tạo chatbot
def create_chatbot(pdf_path, query):
    # Trích xuất và làm sạch văn bản từ PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(pdf_text)

    # Chia văn bản thành các đoạn nhỏ (chunking)
    document_chunks = chunk_text(cleaned_text)
    documents = [Document(page_content=doc) for doc in document_chunks]

    # Tạo embedding cho các đoạn văn bản
    embeddings = HuggingFaceEmbeddings(model_name="hieuduong29", api_key="YOUR_API_KEY")
    vector_store = store_in_faiss(documents, embeddings)

    # Truy vấn và trả về câu trả lời từ mô hình Ollama
    response = query_ollama_model(query)
    if response:
        print(f"Trả lời từ mô hình: {response['choices'][0]['message']['content']}")
    else:
        print("Không thể nhận câu trả lời từ Ollama.")

# Sử dụng chatbot
create_chatbot("QUYCHE_HUST2023.pdf", "Khi nào sinh viên bị đuổi học?")
