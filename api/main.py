# api/main.py

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import config  # 루트의 config 패키지 임포트

# [수정] 'src' 패키지에서 클래스 임포트
from src.services.rag_agent import RagAgent
from src.storage.elastic_store import ElasticStore
from src.clients.ollama_client import OllamaClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 모든 의존성(클래스) 로드
    print("--- 1. Loading ElasticStore (ES Client + Embedding Models)...")
    store = ElasticStore(
        host=config.ES_HOST,
        api_id=config.ES_ID,
        api_key=config.ES_API_KEY,
        index_name=config.ES_INDEX_NAME,
        embedding_model_name=config.ES_EMBEDDING_MODEL  # config에서 모델 이름 주입
    )

    print("--- 2. Loading OllamaClient...")
    ollama = OllamaClient(
        host=config.OLLAMA_HOST,
        model=config.OLLAMA_MODEL
    )

    print("--- 3. Loading RagAgent...")
    # 의존성 주입 (DI)
    agent = RagAgent(vector_store=store, llm_client=ollama)

    app.state.agent = agent
    print("--- 4. RAG API ready. ---")
    yield
    print("--- RAG API shutting down. ---")


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


def get_agent() -> RagAgent:
    """FastAPI 의존성 주입용 함수"""
    return app.state.agent


@app.post("/query")
async def handle_query(request: QueryRequest, agent: RagAgent = Depends(get_agent)):
    """사용자의 질문을 받아 RAG 파이프라인을 실행하고 답변을 반환합니다."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question field cannot be empty.")

    try:
        answer = agent.query(request.question)  # 에이전트의 메서드 호출
        if answer is None:
            raise HTTPException(status_code=500, detail="Failed to generate answer from RAG pipeline.")
        return {"answer": answer}

    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


if __name__ == "__main__":
    # 이 파일을 직접 실행하지 않고, 터미널에서 'uvicorn'으로 실행합니다.
    # 예: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    print("FastAPI 서버를 실행하려면 터미널에서 uvicorn 명령어를 사용하세요:")
    print("uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")