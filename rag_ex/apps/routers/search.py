import asyncio
import time
from fastapi import APIRouter, HTTPException, status

from models.request import SearchRequest
from models.response import SearchResponse
from models.state import SearchResult
from stores.vector_store import elasticsearch_store
from common.config import settings


request_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

router = APIRouter(prefix="/search", tags=["search"])
    
@router.post("/vector", response_model=SearchResponse)
async def vector_search_request(request: SearchRequest):
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
        
@router.post("/keyword", response_model=SearchResponse)
async def keyword_search_request(request: SearchRequest):
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
        
@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search_request(request: SearchRequest):
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