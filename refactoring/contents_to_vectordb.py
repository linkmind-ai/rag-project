import os
import json

from elastic_search import ElasticSearchIndexer
from FlagEmbedding import FlagReranker
from chunker import *
from config import VectorDBConfig


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
        self.reranker = FlagReranker(config.reranker_model_name)
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
    config = VectorDBConfig.from_env_and_file("refactoring/config.json")
    pipeline = ESPipeline(config)
    
    # 🔥 여기에 delete 선택 로직을 둘 수 있음
    user_input = input("ES 인덱스를 모두 삭제할까요? (y/N): ").lower()
    if user_input == "y":
        pipeline.indexer.delete_all_documents()
        
    pipeline.run(chunk_size=800, chunk_overlap=50)