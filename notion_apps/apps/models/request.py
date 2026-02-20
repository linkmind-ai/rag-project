from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


#  sources 필드 타입 안정성 확보를 위한 전용 모델 분리
class SourceItem(BaseModel):
    """응답 근거 문서 모델"""
    doc_id: str = Field(..., description="문서 ID")
    content: str = Field(..., description="근거 내용")
    score: float = Field(..., description="유사도 스코어")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)  #  응답 모델은 불변으로


class QueryRequest(BaseModel):
    """API request 모델"""
    #  uuid1 → uuid4로 변경 (보안: MAC 주소 노출 방지)
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="세션 ID"
    )
    query: str = Field(..., description="사용자 질문", min_length=1)
    use_history: bool = Field(default=True, description="대화 이력 사용 여부")

    model_config = ConfigDict(frozen=False)


class AddDocumentRequest(BaseModel):
    """문서 추가 요청 모델"""
    content: str = Field(..., description="문서 내용", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class SearchRequest(BaseModel):
    """검색 요청 모델"""
    query: str = Field(..., description="사용자 질의")
    top_k: int = Field(default=5, description="top_k 개수", ge=1, le=100)  #  범위 제한 추가
    #  filter → metadata_filter로 변경 (Python 내장 함수명 충돌 방지)
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="메타데이터 필터"
    )

    model_config = ConfigDict(frozen=False)


class NotionPageRequest(BaseModel):
    """노션 페이지 가져오기 요청 모델"""
    page_id: str = Field(..., description="노션 페이지 ID", min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    recursive: bool = Field(default=True, description="하위 블록 재귀 분할")

    model_config = ConfigDict(frozen=False)

    #  노션 페이지 ID 형식 검증 추가 (하이픈 제거 후 32자리)
    @field_validator("page_id")
    @classmethod
    def validate_page_id(cls, v: str) -> str:
        cleaned = v.replace("-", "")
        if len(cleaned) != 32:
            raise ValueError("유효하지 않은 노션 페이지 ID 형식입니다")
        return v
