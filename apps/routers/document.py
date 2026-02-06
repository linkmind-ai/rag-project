import asyncio

from common.config import settings
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from models.request import AddDocumentRequest
from models.response import AddDocumentResponse, FileUploadResponse
from models.state import Document
from stores.vector_store import elasticsearch_store
from utils.file_processor import file_processor

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(prefix="/document", tags=["document"])


@router.post("/add", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """문서 추가 엔드포인트"""
    async with request_semaphore:
        try:
            document = Document(content=request.content, metadata=request.metadata)

            ids = await elasticsearch_store.add_documents([document])

            return AddDocumentResponse(
                success=True, message="문서 추가 성공", document_ids=ids
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 추가 오류 발생: {e!s}",
            ) from e


@router.post("/add_batch", response_model=AddDocumentResponse)
async def add_documents_batch(
    documents: list[AddDocumentRequest],
) -> AddDocumentResponse:
    """문서 일괄 추가"""

    async with request_semaphore:
        try:
            docs = [
                Document(content=doc.content, metadata=doc.metadata)
                for doc in documents
            ]

            ids = await elasticsearch_store.add_documents(docs)

            return AddDocumentResponse(
                success=True,
                message=f"{len(ids)}개 문서 일괄 추가 성공",
                document_ids=ids,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 추가 오류 발생: {e!s}",
            ) from e


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...), metadata: str = Form(default="{}")
) -> FileUploadResponse:
    """파일 업로드 엔드포인트"""

    async with request_semaphore:
        try:
            import json

            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                metadata_dict = {}

            content = await file.read()

            is_valid, error_msg = file_processor.validate_file(
                file.filename, len(content)
            )
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
                )

            file_path = await file_processor.save_file(file.filename, content)

            documents = await file_processor.process_file(file_path)

            for doc in documents:
                doc.metadata.upload(metadata_dict)

            document_ids = await elasticsearch_store.add_documents(documents)

            return FileUploadResponse(
                success=True,
                message="파일 업로드 성공",
                filename=file.filename,
                document_ids=document_ids,
                chunks_count=len(document_ids),
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"파일 업로드 오류 발생: {e!s}",
            ) from e


@router.get("/count")
async def get_document_count() -> dict[str, int | str]:
    """문서 수량 카운트"""
    try:
        count = await elasticsearch_store.get_document_count()

        return {"count": count, "index": settings.ELASTICSEARCH_INDEX}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 수 조회 오류 발생: {e!s}",
        ) from e


@router.delete("/{doc_id}")
async def delete_documents(doc_id: str) -> dict[str, bool | str]:
    """문서 삭제"""
    async with request_semaphore:
        try:
            success = await elasticsearch_store.delete_document(doc_id)
            if success:
                return {"success": True, "message": f"문서 {doc_id}가 삭제되었습니다."}
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="문서를 찾을 수 없습니다.",
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 삭제 오류 발생: {e!s}",
            ) from e
