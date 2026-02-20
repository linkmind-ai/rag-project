from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


#  Message → langchain_core BaseMessage 계열로 대체
# 별도 커스텀 Message 클래스 불필요, BaseMessage 서브클래스 활용
# HumanMessage / AIMessage / SystemMessage 사용


class ChatHistory(BaseModel):
    """대화이력 모델"""
    session_id: str = Field(..., description="세션 ID")
    #  List[BaseMessage]로 변경 - LangGraph add_messages와 호환
    messages: List[BaseMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)  #  arbitrary_types_allowed 추가

    def add_message(self, role: str, content: str) -> None:
        role_map = {
            "human": HumanMessage,
            "user": HumanMessage,
            "ai": AIMessage,
            "assistant": AIMessage,
            "system": SystemMessage,
        }
        msg_cls = role_map.get(role, HumanMessage)
        self.messages.append(msg_cls(content=content))
        self.updated_at = datetime.now()

    def get_recent_messages_history(self, limit: int = 10) -> List[BaseMessage]:
        return self.messages[-limit:]


class Document(BaseModel):
    """문서 모델"""
    content: str = Field(..., description="문서 내용")
    #  Dict → dict (Python 3.9+) 또는 명시적 default
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
    documents: List[Document] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    total_hits: int = Field(default=0)

    model_config = ConfigDict(frozen=False)


#  GraphState: TypedDict + Annotated 방식으로 전환 (LangGraph 권장 패턴)
class GraphState(TypedDict):
    """랭그래프 상태 모델"""
    query: str
    #  add_messages reducer 적용 - 상태 병합 시 자동으로 메시지 누적
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    retrieved_docs: List[Document]
    answer: str
    evidence_indices: List[int]
    session_id: str
