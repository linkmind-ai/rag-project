import json
import asyncio
import uuid

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from langchain.schema import BaseMessage, HumanMessage, AIMessage

import apps.langgraph.state.chat_state import ChatState
import apps.langgraph.cache.in_memory import load_state
import apps.models.base import ChatMessage, TokenUsage
import apps.models.request import ChatMessage
import apps.models.response import ChatStateResponse


router = APIRouter(prefix="/api/gai/usr")

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return await chat_service.stream_chat(request)

@router.get("/chat/stream/{session_id}")
async def get_chat_content(session_id: str):

    state: ChatState = await cache.load_state(session_id)

    if not state or len(state.messages) == 0:
        return ChatStateResponse(
            session_id = session_id,
            messages = [],
            token_usage = TokenUsage()
        )
    def to_msg(m: BaseMessage):
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage) and m.content.startswith("[요약]"):
            role = "summary"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = getattr(m, "role", "assistant")
        return ChatMessage(role = role, content = m.content)
    
    converted_messages = [to_msg(m) for m in state.messages]

    usage = TokenUsage(
        input_tokens = state.token_usage.get("input_tokens", 0),
        output_tokens = state.token_usage.get("output_tokens", 0),
        total_tokens = state.token_usage.get("total_tokens", 0),
    )

    return ChatStateResponse(
        session_id = session_id,
        messages = converted_messages,
        token_usage = usage
    )