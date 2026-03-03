import uuid
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """API query request model."""

    session_id: str = Field(default_factory=lambda: uuid.uuid1().hex)
    query: str = Field(..., min_length=1)
    use_history: bool = Field(default=True)

    model_config = ConfigDict(frozen=False)


class AddDocumentRequest(BaseModel):
    """Document add request model."""

    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(...)
    top_k: int = Field(default=5)
    filters: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("filters", "filter"),
    )

    model_config = ConfigDict(frozen=False)


class NotionPageRequest(BaseModel):
    """Notion import request model."""

    page_id: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recursive: bool = Field(default=True)

    model_config = ConfigDict(frozen=False)


class FeedbackRequest(BaseModel):
    """Explicit feedback request model for profile updates."""

    session_id: str = Field(..., min_length=1)
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_text: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    query: str | None = Field(default=None)
    answer: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=False)
