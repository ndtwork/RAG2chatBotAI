from langchain.schema import Document
from huggingface_hub import login
from langchain.vectorstores import FAISS
import pdfplumber
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


login(token="sk-6182325efac04235aec669b4854dfc31")  # Đăng nhập với API key của Hugging Face

embeddings = HuggingFaceEmbeddings(model_name="hieuduong29")  # Đảm bảo rằng bạn cung cấp tên mô hình


# Trích xuất nội dung từ PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:  # Sử dụng pdfplumber để trích xuất
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

pdf_text = extract_text_from_pdf("c:/Users/HP/Documents/Code/AIchatbot/QUYCHE_HUST2023.pdf")
print(pdf_text)

# Làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Xóa khoảng trắng thừa
    text = re.sub(r'\n+', '\n', text)  # Xóa dòng trống
    return text.strip()

cleaned_text = clean_text(pdf_text)
print(cleaned_text)

# Chia văn bản thành các đoạn nhỏ (chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_text(cleaned_text)

# Chuyển các đoạn văn bản thành các đối tượng Document
document_objects = [Document(page_content=doc) for doc in documents]

# Lưu trữ vector và tìm kiếm FAISS
vector_store = FAISS.from_documents(document_objects, embeddings)
vector_store.save_local("pdf_data_index")

# Truy vấn dữ liệu
query = "Khi nào thì sinh viên bị đuổi học ?"
results = vector_store.similarity_search(query, k=3)

for res in results:
    print(res.page_content)
