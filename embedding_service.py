from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch

# Khởi tạo FastAPI
app = FastAPI()

# Tải mô hình và tokenizer từ transformers
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Định nghĩa cấu trúc dữ liệu đầu vào
class TextInput(BaseModel):
    text: str

class compareTextInput(BaseModel):
    text1: str
    text2: str

@app.post("/embed/")
async def embed_text(text_input: TextInput):
    text = text_input.text

    # Mã hóa đầu vào thành token
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    # Tạo vector nhúng
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    
    return {"embedding": embeddings}

# Đường dẫn để kiểm tra sức khỏe của dịch vụ
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/compare_text")
async def compare_text(text_input: compareTextInput):
    text1 = text_input.text1
    text2 = text_input.text2

    # Mã hóa đầu vào thành token
    inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)

    # Tạo vector nhúng
    with torch.no_grad():
        outputs1 = model(**inputs1)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).squeeze()

        outputs2 = model(**inputs2)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).squeeze()

    # Tính khoảng cách cosine giữa hai vector nhúng
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=0).item()

    return {"similarity": similarity}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
