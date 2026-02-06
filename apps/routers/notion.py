import asyncio

from common.config import settings
from fastapi import APIRouter, HTTPException, status
from models.request import NotionPageRequest
from models.response import NotionPageResponse
from stores.vector_store import elasticsearch_store
from utils.notion_connector import notion_connector

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(prefix="/notion", tags=["notion"])


@router.post("/import", response_model=NotionPageResponse)
async def import_notion_page(request: NotionPageRequest) -> NotionPageResponse:
    """노션 페이지 가져오기 엔드포인트"""
    async with request_semaphore:
        try:
            if not settings.NOTION_TOKEN:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="노션 API 토큰을 다시 설정하세요.",
                )

            page_id = request.page_id.replace("-", "")

            document = await notion_connector.fetch_page_as_document(
                page_id=page_id,
                recursive=request.recursive,
                additional_metadata=request.metadata,
            )

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="노션 페이지를 찾을 수 없습니다. 페이지 ID와 API 정보를 확인해주세요.",
                )

            document_ids = await elasticsearch_store.add_documents([document])

            page_title = document.metadata.get("page_title", "Untitled")

            return NotionPageResponse(
                success=True,
                message="노션 페이지 가져오기 성공",
                page_id=page_id,
                page_title=page_title,
                document_ids=document_ids,
                chunks_count=len(document_ids),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"노션 페이지 가져오기 오류 발생: {e!s}",
            ) from e
