from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message text")
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class ChatHistory(BaseModel):
    """Session chat history model."""

    session_id: str = Field(..., description="Session id")
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.updated_at = datetime.now()

    def get_recent_messages_history(self, limit: int = 10) -> list[Message]:
        return self.messages[-limit:]


class Document(BaseModel):
    """Retrieved document model."""

    content: str = Field(..., description="Document text")
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = Field(default="", description="Document id")

    model_config = ConfigDict(frozen=False)


class SearchResult(BaseModel):
    """Search result model."""

    doc_id: str = Field(..., description="Document id")
    content: str = Field(..., description="Document text")
    score: float = Field(..., description="Search score")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class RetrievedContext(BaseModel):
    """Retrieved context model."""

    documents: list[Document] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    total_hits: int = Field(default=0)

    model_config = ConfigDict(frozen=False)


class SelfRagScores(BaseModel):
    """Minimal Self-RAG outputs for adaptive retry."""

    utility_score: float = 0.0
    confidence: float = 0.0
    insufficiency_reasons: list[str] = Field(default_factory=list)
    next_query: str = ""

    model_config = ConfigDict(frozen=False)


class GraphState(BaseModel):
    """LangGraph state model."""

    query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Session id")
    chat_history: list[Message] = Field(default_factory=list)
    retrieved_docs: list[Document] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    retrieval_query: str = ""
    session_summary: str = ""
    generation_hints: str = ""
    user_profile: dict[str, Any] = Field(default_factory=dict)
    answer: str = Field(default="")
    selfrag_scores: SelfRagScores = Field(default_factory=SelfRagScores)
    loop_count: int = 0
    max_loops: int = 2
    is_sufficient: bool = False
    next_action: Literal["retry", "finalize"] = "finalize"
    next_query: str = ""
    last_retrieval_query: str = ""

    model_config = ConfigDict(frozen=False)
