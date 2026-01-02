import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class Message(BaseModel):
    """대화메시지 모델"""
    role: str = Field(..., description="메시지 유형 분류")
    content: str = Field(..., description="메시지 내용")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class ChatHistory(BaseModel):
    """대화이력 모델"""
    session_id: str = Field(..., description="세션 ID")
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(frozen=False)

    def add_message(self, role: str, content: str) -> None:
        """메시지 추가"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """최근 메시지 load"""
        return self.messages[-limit:]
    

class Document(BaseModel):
    """문서 모델"""
    content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: str = Field(..., description="문서 ID")

    model_config = ConfigDict(frozen=False)


class SearchResult(BaseModel):
    """엘라스틱서치 검색결과 모델"""
    doc_id: str = Field(..., description="문서 ID")
    content: str = Field(..., description="문서 내용")
    score: float = Field(..., description="검색 스코어")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class RetrievedContext(BaseModel):
    """검색된 컨텍스트 모델"""
    documents: List[Document] = Field(default_factory=list, description="검색된 문서")
    scores: List[float] = Field(default_factory=list, description="검색된 문서 스코어")
    total_hits: int = Field(default=0, description="검색 결과 건수")

    model_config = ConfigDict(frozen=False)


class QueryRequest(BaseModel):
    """API request 모델"""
    session_id: str = Field(default_factory=lambda: uuid.uuid1().hex, description="세션 ID")
    query: str = Field(..., description="사용자 질문", min_length=1)
    use_history: bool = Field(default=True, description="대화 이력 사용 여부")

    model_config = ConfigDict(frozen=False)


class QueryResponse(BaseModel):
    """쿼리 응답 모델"""
    session_id: str = Field(..., description="세션 ID")
    answer: str = Field(..., description="LLM 응답")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="응답 근거 내용과 스코어")
    processing_time: float = Field(..., description="처리 시간")

    model_config = ConfigDict(frozen=False)


class AddDocumentRequest(BaseModel):
    """문서 추가 요청 모델"""
    content: str = Field(..., description="문서 내용", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

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


class GraphState(BaseModel):
    """랭그래프 상태 모델"""
    query: str = Field(..., description="사용자 질의")
    chat_history: List[Message] = Field(default_factory=list, description="이전 대화 이력")
    retrieved_docs: List[Document] = Field(default_factory=list, description="검색된 문서")
    answer: str = Field(default="", description="LLM 응답")
    session_id: str = Field(..., description="세션 ID")

    model_config = ConfigDict(frozen=False)


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="사용자 질의")
    top_k: int = Field(default=5, description="top_k 개수")
    filter: Optional[Dict[str, Any]] = Field(None, description="메타데이터 필터")

    model_config = ConfigDict(frozen=False)


class SearchResponse(BaseModel):
    """검색 응답 모델"""
    results: List[SearchResult] = Field(default_factory=list, description="검색 결과")
    total_hits: int = Field(..., description="검색 건수")
    processing_time: float = Field(..., description="처리 시간")

    model_config = ConfigDict(frozen=False)


class NotionPageRequest(BaseModel):
    """노션 페이지 가져오기 요청 모델"""
    page_id: str = Field(..., description="노션 페이지 ID", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recursive: bool = Field(default=True, description="하위 블록 재귀 분할")

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