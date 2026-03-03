from datetime import datetime
from typing import Any

from models.state import SearchResult
from pydantic import BaseModel, ConfigDict, Field


class QueryResponse(BaseModel):
    """Query response model."""

    session_id: str = Field(...)
    answer: str = Field(...)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    processing_time: float = Field(...)
    meta: dict[str, Any] | None = Field(default=None)

    model_config = ConfigDict(frozen=False)


class AddDocumentResponse(BaseModel):
    """Document add response model."""

    success: bool = Field(...)
    message: str = Field(...)
    document_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=False)


class FileUploadResponse(BaseModel):
    """File upload response model."""

    success: bool = Field(...)
    message: str = Field(...)
    filename: str = Field(...)
    document_ids: list[str] = Field(default_factory=list)
    chunks_count: int = Field(...)

    model_config = ConfigDict(frozen=False)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(...)
    elasticsearch_connected: bool = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)


class SearchResponse(BaseModel):
    """Search response model."""

    results: list[SearchResult] = Field(default_factory=list)
    total_hits: int = Field(...)
    processing_time: float = Field(...)

    model_config = ConfigDict(frozen=False)


class NotionPageResponse(BaseModel):
    """Notion import response model."""

    success: bool = Field(...)
    message: str = Field(...)
    page_id: str = Field(...)
    page_title: str = Field(...)
    document_ids: list[str] = Field(default_factory=list)
    chunks_count: int = Field(...)

    model_config = ConfigDict(frozen=False)
