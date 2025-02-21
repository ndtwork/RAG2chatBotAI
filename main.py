import requests
import json


def query_ollama_model(query, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/generate"  # Cập nhật URL theo endpoint từ Postman
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # In ra toàn bộ phản hồi để kiểm tra cấu trúc
    print(response.json())

    try:
        # Trả về nội dung từ phản hồi nếu có
        result = response.json()
        return result['choices'][0]['message']['content']
    except KeyError as e:
        # Nếu không tìm thấy khóa 'choices', in lỗi và trả về phản hồi gốc
        print(f"KeyError: {e}")
        return result  # Trả về toàn bộ phản hồi để kiểm tra


# Thử nghiệm với câu hỏi
query = "Khi nào sinh viên bị đuổi học?"
result = query_ollama_model(query)
print(result)
