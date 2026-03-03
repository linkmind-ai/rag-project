import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from common.config import settings
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from models.request import FeedbackRequest, QueryRequest
from models.response import QueryResponse
from services.service import rag_service

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    async with request_semaphore:
        start_time = time.time()
        try:
            response = await rag_service.process_query(
                session_id=request.session_id,
                query=request.query,
                use_history=request.use_history,
            )

            processing_time = time.time() - start_time
            sources = [
                {
                    "index": idx,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "is_evidence": True,
                }
                for idx, doc in zip(
                    response["evidence_indices"], response["evidence_docs"], strict=True
                )
            ]
            sources.sort(key=lambda x: x["index"])

            return QueryResponse(
                session_id=request.session_id,
                answer=response["answer"],
                sources=sources,
                processing_time=processing_time,
                meta=response.get("meta"),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query processing failed: {e!s}",
            ) from e


@router.post("/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    async def generate() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': request.session_id})}\n\n"
            async for event in rag_service.process_query_stream(
                session_id=request.session_id,
                query=request.query,
                use_history=request.use_history,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, Any]:
    async with request_semaphore:
        try:
            result = await rag_service.submit_feedback(request.model_dump())
            return result
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feedback processing failed: {e!s}",
            ) from e
