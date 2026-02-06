import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """API request 모델"""

    session_id: str = Field(
        default_factory=lambda: uuid.uuid1().hex, description="세션 ID"
    )
    query: str = Field(..., description="사용자 질문", min_length=1)
    use_history: bool = Field(default=True, description="대화 이력 사용 여부")

    model_config = ConfigDict(frozen=False)


class AddDocumentRequest(BaseModel):
    """문서 추가 요청 모델"""

    content: str = Field(..., description="문서 내용", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class SearchRequest(BaseModel):
    """검색 요청 모델"""

    query: str = Field(..., description="사용자 질의")
    top_k: int = Field(default=5, description="top_k 개수")
    filter: dict[str, Any] | None = Field(None, description="메타데이터 필터")

    model_config = ConfigDict(frozen=False)


class NotionPageRequest(BaseModel):
    """노션 페이지 가져오기 요청 모델"""

    page_id: str = Field(..., description="노션 페이지 ID", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recursive: bool = Field(default=True, description="하위 블록 재귀 분할")

    model_config = ConfigDict(frozen=False)
