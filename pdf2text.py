import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path, output_txt_path):
    # Mở file PDF
    reader = PdfReader(pdf_path)
    text = ""

    # Duyệt qua từng trang và trích xuất văn bản
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Lưu vào file TXT
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(text)

    print(f"✅ Đã trích xuất văn bản và lưu vào {output_txt_path}")

#  Thay đổi đường dẫn file PDF và file TXT tại đây
pdf_file = "QUYCHE_HUST2023.pdf"  # Đường dẫn file PDF
txt_file = "quyche.txt"  # File TXT sẽ lưu kết quả

# Gọi hàm trích xuất
extract_text_from_pdf(pdf_file, txt_file)
