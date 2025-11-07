# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import config
import elasticsearch_manager
import run_hyde_search

app = FastAPI()

# 1. 모델과 ES 클라이언트를 앱 시작 시 한 번만 로드
@app.on_event("startup")
def load_models():
    app.state.es_client = elasticsearch_manager.get_es_client()
    app.state.tokenizer, app.state.model = elasticsearch_manager.get_embedding_model()

    if not app.state.es_client or not app.state.tokenizer:
        raise RuntimeError("필수 모델 또는 ES 클라이언트 로드 실패")

class QueryRequest(BaseModel):
    question: str

# 2. RAG API 엔드포인트 생성
@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        answer = run_hyde_search.run_hyde_pipeline(
            request.question,
            app.state.es_client,
            app.state.tokenizer,
            app.state.model
        )
        if answer is None:
            raise HTTPException(status_code=500, detail="Ollama 또는 ES에서 답변 생성 실패")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)