import json
import asyncio
from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

from prompts.summary_prompt import SUMMARY_PROMPT
from prompts.chat_prompt import BASE_CHAT_PROMPT
from clients.llm_client import CustomLLMClient
from models.request import ChatRequest
from models.response import ChatStateResponse
from langgraph.graph.rag_chat import RagChatGraph
from langgraph.cache.in_memory import InMemoryCache
from services.chat_service import ChatService


semaphore = asyncio.Semaphore(50)

llm_client = CustomLLMClient()

cache = InMemoryCache()
_graph = RagChatGraph(
    llm = llm_client(),
    summary_prompt = SUMMARY_PROMPT,
    base_chat_prompt = BASE_CHAT_PROMPT
)

shared_graph = _graph.build_graph()

chat_service = ChatService(
    graph = shared_graph,
    cache = cache,
    semaphore = semaphore
)

router = APIRouter(prefix="/api/rag")

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return await chat_service.stream_chat(request)

@router.post("/chat/state/{session_id}", response_model=ChatStateResponse)
async def get_chat_state(session_id: str):
    return await chat_service.get_state(session_id)
