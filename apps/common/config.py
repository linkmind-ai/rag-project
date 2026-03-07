from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리 (.env 파일 위치)
PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """설정값 관리"""

    # OLLAMA 설정
    OLLAMA_BASE_URL: str = Field(
        default="https://ollama.nabee.ai.kr/",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    )
    OLLAMA_MODEL: str = Field(default="hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:BF16")

    # 엘라스틱서치 설정
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

    # 단일성 정체감의 장애 현상은 어떻게 나타나는가?
    # 벡터스토어 설정
    # EMBEDDING_MODEL: str = Field(default="")
    # PERSIST_DIR: str = Field(default="")
    # COLLECTION_NAME: str = Field(default="notion_documents")

    # API 설정
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default="8000")
    MAX_CONCURRENT_REQUESTS: int = Field(default=5)

    # RAG 설정
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    TOP_K_RESULTS: int = Field(default=3)
    MAX_HISTORY_LENGTH: int = Field(default=10)

    # 문서 업로드 설정
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024)
    ALLOWED_EXTENSIONS: list = Field(default=["txt", "pdf", "docx", "md"])

    # Cloudflare Access 설정
    CF_ACCESS_CLIENT_ID: str = Field(default="")
    CF_ACCESS_CLIENT_SECRET: str = Field(default="")

    # Groq API 설정 (RAGAS 평가 전용)
    GROQ_API_KEY: str = Field(default="")
    GROQ_API_KEY_2: str = Field(default="")

    # Notion 연동 설정
    NOTION_TOKEN: str = Field(
        default="ntn_S49134845636QN7OizYlyythCTORUXOCvYcp2U19S0P6dy"
    )
    NOTION_VERSION: str = Field(default="2022-06-28")

    # Tavily Search API 설정
    TAVILY_API_KEY: str

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        case_sensitive = True
        extra = "ignore"


settings = Settings()
