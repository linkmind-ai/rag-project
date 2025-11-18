import os
import json
from dotenv import load_dotenv
from dataclasses import dataclass

from elastic_search import ElasticSearchIndexer
from chunker import *


# =========================
# 설정 / 환경 로딩
# =========================

@dataclass
class VectorDBConfig:
    es_host: str
    es_id: str
    es_api_key: str
    es_index: str
    contents_path: str
    txt_path: str
    json_path: str
    vec_dims: int
    keyword_model_name: str
    embedding_model_name: str

    @classmethod
    def from_env_and_file(cls, config_path: str = "template/common/config.json") -> "VectorDBConfig":
        load_dotenv()

        es_host = os.getenv("ES_HOST")
        es_id = os.getenv("ES_ID")
        es_api_key = os.getenv("ES_API_KEY")
        es_index = os.getenv("ES_INDEX_NAME", "vector-test-index")
        contents_path = os.getenv("NOTION_CONTENTS_PATH", ".")
        txt_path = os.getenv("NOTION_TXT_PATH", "notion_page_content.txt")
        json_path = os.getenv("NOTION_JSON_PATH", "notion_page_content.json")
        vec_dims = int(os.getenv("VEC_DIMS", "768"))

        if not es_host or not es_id or not es_api_key:
            raise ValueError("❌ ES_HOST, ES_ID, ES_API_KEY 환경변수를 확인하세요.")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        keyword_model_name = cfg["model"]["keyword"]
        embedding_model_name = cfg["model"]["embedding"]

        return cls(
            es_host=es_host,
            es_id=es_id,
            es_api_key=es_api_key,
            es_index=es_index,
            contents_path=contents_path,
            txt_path=txt_path,
            json_path=json_path,
            vec_dims=vec_dims,
            keyword_model_name=keyword_model_name,
            embedding_model_name=embedding_model_name,
        )


# =========================
# 전체 파이프라인 오케스트레이터
# =========================

class ESPipeline:
    def __init__(self, config: VectorDBConfig) -> None:
        self.config = config
        self.loader = TextLoader(config.contents_path)
        self.noun_extractor = KiwiNounExtractor(config.keyword_model_name)
        self.chunker = Chunker()
        self.embedder = EmbeddingModel(config.embedding_model_name)
        self.indexer = ElasticSearchIndexer(config)

    def run(self, chunk_size: int = 800, chunk_overlap: int = 50) -> None:
        # 1. 텍스트 로드
        text = self.loader.load_text(self.config.txt_path)

        # 2. 페이지 단위로 나누고 키워드 부여
        header_chunks = self.noun_extractor.add_keywords_to_page(
            text,
            top_n=3,
            ngram_range=(1, 1),
        )

        # 3. RecursiveCharacterTextSplitter로 더 잘게 청킹
        chunks = self.chunker.chunk_with_recursive_splitter(
            header_chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 4. ES 인덱싱
        self.indexer.index_documents(chunks, self.embedder)



# =========================
# 엔트리 포인트
# =========================

if __name__ == "__main__":
    config = VectorDBConfig.from_env_and_file("template/common/config.json")
    pipeline = ESPipeline(config)
    pipeline.run(chunk_size=800, chunk_overlap=50)