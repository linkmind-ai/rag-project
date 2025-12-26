from typing import List
from elasticsearch import Elasticsearch
from langchain.schema import Document

from chunker import EmbeddingModel


# =========================
# Elasticsearch 인덱서
# =========================

class ElasticSearchIndexer:
    def __init__(self, config) -> None:
        self.config = config
        self.es = Elasticsearch(
            config.es_host,
            api_key=(config.es_id, config.es_api_key),
        )

    def create_index_if_not_exists(self) -> None:
        if not self.es.indices.exists(index=self.config.es_index):
            self.es.indices.create(
                index=self.config.es_index,
                mappings={
                    "properties": {
                        "content": {"type": "text"},
                        "metadata": {"type": "object", "enabled": True},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.config.vec_dims,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                },
            )

    def index_documents(self, docs: List[Document], embedder: EmbeddingModel) -> None:
        self.create_index_if_not_exists()
        index_name = self.config.es_index

        for i, doc in enumerate(docs):
            content = doc.page_content
            metadata = doc.metadata

            body = {
                "content": content,
                "metadata": metadata,
                "embedding": embedder.embed_text(content),
            }
            self.es.index(index=index_name, id=f"chunk-{i}", document=body)

        print("✅ 문서와 임베딩이 Elasticsearch에 저장되었습니다.")
        
    def delete_all_documents(self):
        print(f"⚠️ 인덱스 '{self.config.es_index}' 내 모든 문서를 삭제합니다...")
        response = self.es.delete_by_query(
            index=self.config.es_index,
            body={"query": {"match_all": {}}},
        )
        print("🗑 삭제 완료:", response)