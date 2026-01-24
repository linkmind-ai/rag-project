from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import query, document, search, session, notion, system
from stores.vector_store import elasticsearch_store
from utils.notion_connector import notion_connector
from services.service import rag_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """어플리케이션 수명 관리"""
    await elasticsearch_store.initialize()
    await rag_service.initialize()
    print("RAG시스템 초기화 완료")

    yield

    await elasticsearch_store.close()
    await notion_connector.close()
    print("RAG 시스템 종료")

app = FastAPI(
    title="RAG System API",
    description="멀티턴 RAG 시스템",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(query.router)
app.include_router(document.router)
app.include_router(search.router)
app.include_router(session.router)
app.include_router(notion.router)