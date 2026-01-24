from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from models.state import SearchResult


class QueryResponse(BaseModel):
    """쿼리 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    answer: str = Field(..., description="LLM 응답")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="응답 근거 내용과 스코어")
    processing_time: float = Field(..., description="처리 시간")

    model_config = ConfigDict(frozen=False)


class AddDocumentResponse(BaseModel):
    """문서 추가 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    document_ids: List[str] = Field(default_factory=list, description="문서 ID")

    model_config = ConfigDict(frozen=False)


class FileUploadResponse(BaseModel):
    """파일 업로드 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    filename: str = Field(..., description="파일명")
    document_ids: List[str] = Field(default_factory=list, description="업로드 문서 ID")
    chunks_count: int = Field(..., description="생성된 청크 수")

    model_config = ConfigDict(frozen=False)


class HealthResponse(BaseModel):
    """헬스 체크 응답 모델"""
    status: str = Field(..., description="서비스 상태")
    elasticsearch_connected: bool = Field(..., description="엘라스틱서치 연결 상태")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[SearchResult] = Field(default_factory=list, description="검색 결과")
    total_hits: int = Field(..., description="검색 건수")
    processing_time: float = Field(..., description="처리 시간")

    model_config = ConfigDict(frozen=False)


class NotionPageResponse(BaseModel):
    """노션 페이지 가져오기 응답 모델"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    page_id: str = Field(..., description="노션 페이지 ID")
    page_title: str = Field(..., description="노션 페이지 제목")
    document_ids: List[str] = Field(default_factory=list, description="생성 문서 ID")
    chunks_count: int = Field(..., description="생성된 청크 수")

    model_config = ConfigDict(frozen=False)