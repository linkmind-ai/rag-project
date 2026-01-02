import asyncio
import time
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from models import (
    QueryRequest, QueryResponse,
    AddDocumentRequest, AddDocumentResponse,
    Document, HealthResponse, FileUploadResponse,
    SearchRequest, SearchResponse, SearchResult,
    NotionPageRequest, NotionPageResponse
)
from vector_store import elasticsearch_store
from memory_sotre import memory_store
from service import rag_service
from file_processor import file_processor
from notion_connector import notion_connector
from config import settings

request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """어플리케이션 수명 관리"""
    await elasticsearch_store.initialize()
    await rag_graph.initialize()
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

@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    es_connected = await elasticsearch_store.health_check()
    return HealthResponse(
        status="running",
        elasticsearch_connected=es_connected
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """통신 상태 점검"""
    es_connected = await elasticsearch_store.health_check()
    return HealthResponse(
        status="healthy" if es_connected else "degraded",
        elasticsearch_connected=es_connected
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """질의응답 엔드포인트"""
    async with request_semaphore:
        start_time = time.time()

        try:
            answer = await rag_service.process_query(
                session_id=request.session_id,
                query=request.query,
                use_history=request.use_history
            )

            processing_time = time.time() - start_time

            sources_data = await rag_service.get_sources(request.query)

            return QueryResponse(
                session_id=request.session_id,
                answer=answer,
                sources=sources_data["sources"],
                processing_time=processing_time
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"쿼리 처리 오류 발생: {str(e)}"
            )
        
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    질의응답 스트리밍 엔드포인트
    
    이벤트 타입:
    1. retrieve_start: 문서 검색 시작
    2. retrieve_end: 문서 검색 완료
    3. generate_start: 답변 생성 시작
    4. generate_end: 답변 생성 완료
    5. done: graph 처리 완료
    """
    async def generate():
        try:
            yield f"data: {json.dumps({'type': 'session_id', 'session_id': request.session_id})}\n\n"

            async for event in rag_service.process_query_stream(
                session_id=request.session_id,
                query=request.query,
                use_history=request.use_history
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
            "X-Accel-Buffering": "no"
        }
    )
        
@app.post("/documents/add", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest):
    """문서 추가 엔드포인트"""
    async with request_semaphore:
        try:
            document = Document(
                content=request.content,
                metadata=request.metadata
            )

            ids = await elasticsearch_store.add_documents([document])

            return AddDocumentResponse(
                success=True,
                message="문서 추가 성공",
                document_ids=ids
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 추가 오류 발생: {str(e)}"
            )
        
@app.post("/documents/add_batch", response_model=AddDocumentResponse)
async def add_documents_batch(documents: List[AddDocumentRequest]):
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
                document_ids=ids
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 추가 오류 발생: {str(e)}"
            )
        
@app.post("/documents/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}")
):
    """파일 업로드 엔드포인트"""

    async with request_semaphore:
        try:
            import json

            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                metadata_dict = {}

            content = await file.read()

            is_valid, error_msg = file_processor.validate_file(file.filename, len(content))
            if not is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            
            file_path = await file_processor.save_file(file.filename, content)

            documents = await file_processor.process_file(file_path)

            for doc in documents:
                doc.metadata.upload(metadata_dict)

            document_ids = await elasticsearch_store.add_documents(documents)

            return FileUploadResponse(
                success=True,
                message=f"파일 업로드 성공",
                filename=file.filename,
                document_ids=document_ids,
                chunks_count=len(document_ids)
            )
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"파일 업로드 오류 발생: {str(e)}"
            )
        
@app.post("/notion/import", response_model=NotionPageResponse)
async def import_notion_page(request: NotionPageRequest):
    """노션 페이지 가져오기 엔드포인트"""
    async with request_semaphore:
        try:
            if not settings.NOTION_TOKEN:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"노션 API 토큰을 다시 설정하세요."
                )

            page_id = request.page_id.replace("-", "")

            document = await notion_connector.fetch_page_as_document(
                page_id=page_id,
                recursive=request.recursive,
                additional_metadata=request.metadata
            )

            if not document:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"노션 페이지를 찾을 수 없습니다. 페이지 ID와 API 정보를 확인해주세요."
                )
            
            document_ids = await elasticsearch_store.add_documents([document])

            page_title = document.metadata.get("page_title", "Untitled")
                      
            return NotionPageResponse(
                success=True,
                message="노션 페이지 가져오기 성공",
                page_id=page_id,
                page_title=page_title,
                document_ids=document_ids,
                chunks_count=len(document_ids)
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"노션 페이지 가져오기 오류 발생: {str(e)}"
            )
        
@app.post("/search/vector", response_model=SearchResponse)
async def vector_search(request: SearchRequest):
    """벡터 유사도 검색"""
    async with request_semaphore:
        start_time = time.time()

        try:
            context = await elasticsearch_store.similarity_search(
                query=request.query,
                k=request.top_k,
                filter=request.filters
            )

            processing_time = time.time() - start_time

            results = [
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata 
                )
                for doc, score in zip(context.documents, context.scores)
            ]

            return SearchResponse(
                results=results,
                total_hits=context.total_hits,
                processing_time=processing_time
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"검색 오류 발생: {str(e)}"
            )
        
@app.post("/search/keyword", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    """키워드 기반 검색"""
    async with request_semaphore:
        start_time = time.time()

        try:
            context = await elasticsearch_store.kwyword_search(
                query=request.query,
                k=request.top_k,
                filters=request.filters
            )

            processing_time = time.time() - start_time

            results = [
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata
                )
                for doc, score in zip(context.documents, context.scores)
            ]

            return SearchResponse(
                results=results,
                total_hits=context.total_hits,
                processing_time=processing_time
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"검색 오류 발생: {str(e)}"
            )
        
@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """하이브리드 검색"""
    async with request_semaphore:
        start_time = time.time()

        try:
            context = await elasticsearch_store.hybrid_search(
                query=request.query,
                k=request.top_k,
                filters=request.filters
            )

            processing_time = time.time() - start_time

            results = [
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata
                )
                for doc, score in zip(context.documents, context.scores)
            ]

            return SearchResponse(
                results=results,
                total_hits=context.total_hits,
                processing_time=processing_time
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"검색 오류 발생: {str(e)}"
            )
        
@app.get("/documents/count")
async def get_document_count():
    """문서 수량 카운트"""
    try:
        count = await elasticsearch_store.get_document_count()
            
        return {
            "count": count,
            "index": settings.ELASTICSEARCH_INDEX
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 수 조회 오류 발생: {str(e)}"
        )
    
@app.delete("/documents/{doc_id}")
async def delete_documents(doc_id: str):
    """문서 삭제"""
    async with request_semaphore:
        try:
            success = await elasticsearch_store.delete_document(doc_id)
            if success:
                return {
                    "success": True,
                    "message": f"문서 {doc_id}가 삭제되었습니다."
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"문서를 찾을 수 없습니다."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"문서 삭제 오류 발생: {str(e)}"
            )
        
@app.get("/sessions")
async def get_sessions():
    """세션 목록 조회"""
    try:
        sessions = await memory_store.get_all_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 조회 오류 발생: {str(e)}"
        )
    
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """특정 세션 대화 이력 조회"""
    try:
        messages = await memory_store.get_recent_messages(session_id)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ],
            "count": len(messages)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 이력 조회 오류 발생: {str(e)}"
        )
    
@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """특정 세션 대화 이력 삭제"""
    async with request_semaphore:
        try:
            success = await elasticsearch_store.clear_history(session_id)
            if success:
                return {
                    "success": True,
                    "message": f"세션 {session_id}의 이력이 삭제되었습니다."
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"세션을 찾을 수 없습니다."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"세션 삭제 오류 발생: {str(e)}"
            )

@app.delete("/index/clear")
async def clear_index():
    """엘라스틱서치 인덱스 초기화"""
    async with request_semaphore:
        try:
            await elasticsearch_store.clear()
            return {
                "success": True,
                "message": "인덱스가 초기화 되었습니다."
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"인덱스 초기화 오류 발생: {str(e)}"
            )
        

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )