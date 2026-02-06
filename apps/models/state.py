from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """대화메시지 모델"""

    role: str = Field(..., description="메시지 유형 분류")
    content: str = Field(..., description="메시지 내용")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class ChatHistory(BaseModel):
    """대화이력 모델"""

    session_id: str = Field(..., description="세션 ID")
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)

    def add_message(self, role: str, content: str) -> None:
        """메시지 추가"""
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()

    def get_recent_messages_history(self, limit: int = 10) -> list[Message]:
        """최근 메시지 load"""
        return self.messages[-limit:]


class Document(BaseModel):
    """문서 모델"""

    content: str = Field(..., description="문서 내용")
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = Field(..., description="문서 ID")

    model_config = ConfigDict(frozen=False)


class SearchResult(BaseModel):
    """엘라스틱서치 검색결과 모델"""

    doc_id: str = Field(..., description="문서 ID")
    content: str = Field(..., description="문서 내용")
    score: float = Field(..., description="검색 스코어")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class RetrievedContext(BaseModel):
    """검색된 컨텍스트 모델"""

    documents: list[Document] = Field(default_factory=list, description="검색된 문서")
    scores: list[float] = Field(default_factory=list, description="검색된 문서 스코어")
    total_hits: int = Field(default=0, description="검색 결과 건수")

    model_config = ConfigDict(frozen=False)


class GraphState(BaseModel):
    """랭그래프 상태 모델"""

    query: str = Field(..., description="사용자 질의")
    chat_history: list[Message] = Field(
        default_factory=list, description="이전 대화 이력"
    )
    retrieved_docs: list[Document] = Field(
        default_factory=list, description="검색된 문서"
    )
    answer: str = Field(default="", description="LLM 응답")
    evidence_indices: list[int] = Field(
        default_factory=list, description="근거 문서 인덱스"
    )
    session_id: str = Field(..., description="세션 ID")

    model_config = ConfigDict(frozen=False)
