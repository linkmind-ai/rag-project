import asyncio
from fastapi import APIRouter, HTTPException, status

from models.response import HealthResponse
from stores.vector_store import elasticsearch_store
from common.config import settings

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(tags=["system"])


@router.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    es_connected = await elasticsearch_store.health_check()
    return HealthResponse(status="running", elasticsearch_connected=es_connected)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """통신 상태 점검"""
    es_connected = await elasticsearch_store.health_check()
    return HealthResponse(
        status="healthy" if es_connected else "degraded",
        elasticsearch_connected=es_connected,
    )


@router.delete("/index/clear")
async def clear_index():
    """엘라스틱서치 인덱스 초기화"""
    async with request_semaphore:
        try:
            await elasticsearch_store.clear()
            return {"success": True, "message": "인덱스가 초기화 되었습니다."}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"인덱스 초기화 오류 발생: {str(e)}",
            )
