from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings."""

    # LLM
    OLLAMA_BASE_URL: str = Field(
        default="https://ollama.nabee.ai.kr/",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    OLLAMA_MODEL: str = Field(default="hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:BF16")

    # Elasticsearch
    ELASTICSEARCH_URL: str = Field(
        default="https://es.nabee.ai.kr/",
        validation_alias=AliasChoices("ELASTICSEARCH_URL", "ES_HOST"),
    )
    ELASTICSEARCH_INDEX: str = Field(
        default="vector-test-index",
        validation_alias=AliasChoices("ELASTICSEARCH_INDEX", "ES_INDEX"),
    )
    ELASTICSEARCH_USER: str = Field(
        default="TfsG65sBiLW-D9U8Xu4q",
        validation_alias=AliasChoices("ELASTICSEARCH_USER", "ES_ID"),
    )
    ELASTICSEARCH_PASSWORD: str = Field(
        default="HRo4sn6jiCEt6qSE3uY5xg",
        validation_alias=AliasChoices("ELASTICSEARCH_PASSWORD", "ES_API_KEY"),
    )

    EMBEDDING_MODEL: str = Field(default="bge-m3:latest")
    EMBEDDING_DIM: int = Field(
        default=1024,
        validation_alias=AliasChoices("EMBEDDING_DIM", "VEC_DIMS"),
    )

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    MAX_CONCURRENT_REQUESTS: int = Field(default=5)

    # RAG base
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    TOP_K_RESULTS: int = Field(default=3)
    MAX_HISTORY_LENGTH: int = Field(default=10)

    # PersonaRAG + Self-RAG
    ROUTING_GATE_MODE: str = Field(default="rule_llm_fallback")
    SELF_RAG_MAX_LOOPS: int = Field(default=2)
    SELF_RAG_SEGMENT_MODE: str = Field(default="sentence")
    SELF_RAG_PROFILE: str = Field(default="balanced")

    MIN_UTILITY: float = Field(default=3.5)
    MIN_AVG_REL: float = Field(default=0.55)
    MAX_NO_SUPPORT_RATIO: float = Field(default=0.30)
    MAX_PARTIAL_OR_NO_RATIO: float = Field(default=0.55)
    MIN_FULL_SUPPORT_RATIO_HIGH_RISK: float = Field(default=0.75)

    SELF_RAG_LOOP0_WEIGHTS: tuple[float, float, float] = Field(default=(0.4, 0.4, 0.2))
    SELF_RAG_LOOP1_WEIGHTS: tuple[float, float, float] = Field(default=(0.3, 0.5, 0.2))
    SELF_RAG_LOOP2_WEIGHTS: tuple[float, float, float] = Field(default=(0.2, 0.6, 0.2))

    SELF_RAG_LOOP0_DELTA: float = Field(default=0.55)
    SELF_RAG_LOOP1_DELTA: float = Field(default=0.45)
    SELF_RAG_LOOP2_DELTA: float = Field(default=0.35)

    # Upload
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024)
    ALLOWED_EXTENSIONS: list[str] = Field(default=["txt", "pdf", "docx", "md"])

    # Notion
    NOTION_TOKEN: str = Field(
        default="ntn_S49134845636QN7OizYlyythCTORUXOCvYcp2U19S0P6dy"
    )
    NOTION_VERSION: str = Field(default="2022-06-28")

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        case_sensitive = True
        extra = "ignore"


settings = Settings()
