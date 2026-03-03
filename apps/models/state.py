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


class RouteDecision(BaseModel):
    """Step-1 routing decision."""

    task_type: Literal["creative", "conversational", "factual", "ambiguous"] = (
        "ambiguous"
    )
    risk_level: Literal["low", "high"] = "low"
    retrieval_policy: Literal["minimal", "forced", "adaptive"] = "adaptive"
    used_llm_fallback: bool = False

    model_config = ConfigDict(frozen=False)


class EvidenceBundleE0(BaseModel):
    """Initial personalized evidence bundle from PersonaRAG."""

    top_docs: list[Document] = Field(default_factory=list)
    doc_summaries: list[dict[str, Any]] = Field(default_factory=list)
    citations_meta: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


class GlobalMessagePoolM0(BaseModel):
    """Global message pool shared by PersonaRAG agents."""

    profile_snapshot: dict[str, Any] = Field(default_factory=dict)
    session_summary: str = ""
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    rerank_notes: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


class SelfRagScores(BaseModel):
    """Self-RAG critique outputs and aggregates."""

    retrieve_decisions: list[str] = Field(default_factory=list)
    rel_scores: list[float] = Field(default_factory=list)
    support_scores: list[dict[str, float]] = Field(default_factory=list)
    utility_score: float = 0.0
    avg_isrel: float = 0.0
    no_support_ratio: float = 0.0
    partial_or_no_support_ratio: float = 0.0
    full_support_ratio: float = 0.0
    objective_score: float = 0.0
    insufficiency_reasons: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


class GraphState(BaseModel):
    """LangGraph state model."""

    query: str = Field(..., description="User query")
    chat_history: list[Message] = Field(default_factory=list)
    retrieved_docs: list[Document] = Field(default_factory=list)
    answer: str = Field(default="")
    evidence_indices: list[int] = Field(default_factory=list)
    session_id: str = Field(..., description="Session id")

    route: RouteDecision = Field(default_factory=RouteDecision)
    persona_bundle: EvidenceBundleE0 = Field(default_factory=EvidenceBundleE0)
    global_message_pool: GlobalMessagePoolM0 = Field(
        default_factory=GlobalMessagePoolM0
    )

    draft_answer: str = ""
    final_answer: str = ""
    selfrag_scores: SelfRagScores = Field(default_factory=SelfRagScores)

    loop_count: int = 0
    max_loops: int = 2
    strictness_level: int = 0

    transparency: dict[str, Any] = Field(default_factory=dict)
    is_sufficient: bool = False
    next_action: Literal["reinforce", "finalize"] = "finalize"

    answer_candidates: list[dict[str, Any]] = Field(default_factory=list)
    last_retrieval_query: str = ""

    model_config = ConfigDict(frozen=False)
