from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """설정값 관리"""

    # OLLAMA 설정
    OLLAMA_BASE_URL: str = Field(default="")
    OLLAMA_MODEL: str = Field(default="")

    # 엘라스틱서치 설정
    ELASTICSEARCH_URL: str = Field(default="")
    ELASTICSEARCH_INDEX: str = Field(default="")
    ELASTICSEARCH_USER: str = Field(default="")
    ELASTICSEARCH_PASSWORD: str = Field(default="")
    EMBEDDING_MODEL: str = Field(default="")
    EMBEDDING_DIM: int = Field(default=1024)

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

    # GeAR 설정
    USE_GEAR_RETRIEVAL: bool = Field(
        default=False, description="GeAR 기반 검색 사용 여부"
    )
    GEAR_MAX_STEPS: int = Field(default=3, description="GeAR 검색 최대 횟수")
    GEAR_ENABLE_GRAPH_EXPANSION: bool = Field(
        default=True, description="그래프 확장 사용 여부"
    )

    # 문서 업로드 설정
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024)
    ALLOWED_EXTENSIONS: list = Field(default=["txt", "pdf", "docx", "md"])

    # Notion 연동 설정
    NOTION_TOKEN: str = Field(default="")
    NOTION_VERSION: str = Field(default="2022-06-28")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
