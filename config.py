import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass


@dataclass
class NotionConfig:
    token: str
    version: str
    contents_path: str
    txt_path: str
    json_path: str

    @classmethod
    def from_env(cls) -> "NotionConfig":
        """환경 변수에서 설정을 로드."""
        load_dotenv()

        token = os.getenv("NOTION_TOKEN")
        if not token:
            raise ValueError("❌ 환경변수 NOTION_TOKEN이 설정되어 있지 않습니다. .env 파일을 확인하세요.")

        version = os.getenv("NOTION_VERSION", "2022-06-28")
        contents_path = os.getenv("NOTION_CONTENTS_PATH", ".")
        txt_path = os.getenv("NOTION_TXT_PATH", "notion_page_content.txt")
        json_path = os.getenv("NOTION_JSON_PATH", "notion_page_content.json")

        return cls(
            token=token,
            version=version,
            contents_path=contents_path,
            txt_path=txt_path,
            json_path=json_path,
        )


@dataclass
class VectorDBConfig:
    es_host: str
    es_id: str
    es_api_key: str
    es_index: str
    ollama_host: str
    contents_path: str
    txt_path: str
    json_path: str
    vec_dims: int
    keyword_model_name: str
    embedding_model_name: str
    reranker_model_name: str
    generation_model_name: str

    @classmethod
    def from_env_and_file(cls) -> "VectorDBConfig":
        load_dotenv()

        es_host = os.getenv("ES_HOST")
        es_id = os.getenv("ES_ID")
        es_api_key = os.getenv("ES_API_KEY")
        es_index = os.getenv("ES_INDEX_NAME", "vector-test-index")
        contents_path = os.getenv("NOTION_CONTENTS_PATH", ".")
        txt_path = os.getenv("NOTION_TXT_PATH", "refactoring/documents/notion_page_content.txt")
        json_path = os.getenv("NOTION_JSON_PATH", "refactoring/documents/notion_page_content.json")
        vec_dims = int(os.getenv("VEC_DIMS", "768"))
        ollama_host = os.getenv("OLLAMA_HOST")

        if not es_host or not es_id or not es_api_key:
            raise ValueError("❌ ES_HOST, ES_ID, ES_API_KEY 환경변수를 확인하세요.")

        keyword_model_name = os.getenv("KEYWORD_MODEL")
        embedding_model_name = os.getenv("EMBEDDING_MODEL")
        reranker_model_name = os.getenv("RERANKER_MODEL")
        generation_model_name = os.getenv("GENERATION_MODEL")

        return cls(
            es_host=es_host,
            es_id=es_id,
            es_api_key=es_api_key,
            es_index=es_index,
            ollama_host=ollama_host,
            contents_path=contents_path,
            txt_path=txt_path,
            json_path=json_path,
            vec_dims=vec_dims,
            keyword_model_name=keyword_model_name,
            embedding_model_name=embedding_model_name,
            reranker_model_name=reranker_model_name,
            generation_model_name=generation_model_name
        )