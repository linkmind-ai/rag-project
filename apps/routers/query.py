import asyncio
import json
import time
from collections.abc import AsyncGenerator

from common.config import settings
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from models.request import QueryRequest
from models.response import QueryResponse
from services.service import rag_service

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """질의응답 엔드포인트"""
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
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"쿼리 처리 오류 발생: {e!s}",
            ) from e


@router.post("/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """
    질의응답 스트리밍 엔드포인트

    이벤트 타입:
    1. retrieve_start: 문서 검색 시작
    2. retrieve_end: 문서 검색 완료
    3. generate_start: 답변 생성 시작
    4. stream: 답변 내용 스트리밍
    5. generate_end: 답변 생성 완료
    6. identify_evidence_start: 답변 근거 청크 식별 시작
    7. identify_evidence_end: 답변 근거 청크 식별 완료
    8. done: graph 처리 완료
    """

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
            error_msg = f"data: {json.sumps({'type': 'error', 'error': str(e)})}\n\n"
            yield error_msg

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
