from elasticsearch import Elasticsearch
from apps.common.property.config import ES_HOST, ES_ID, ES_API_KEY, VEC_DIMS, ES_INDEX
from typing import List, Dict, Any


class EmbeddingClient:
    def __init__(self):
        self.es = Elasticsearch(
            ES_HOST,
            api_key=(ES_ID, ES_API_KEY)
        )

        self.index_name = ES_INDEX
        self.dim = VEC_DIMS

        # 인덱스 없으면 생성
        self._ensure_index()

    def _ensure_index(self):
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "source": {"type": "keyword"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": self.dim,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    }
                }
            )

    def upsert_chunk(self, chunk_id: str, content: str, source: str, embedding: List[float]):
        doc = {
            "content": content,
            "source": source,
            "embedding": embedding
        }

        self.es.index(
            index=self.index_name,
            id=chunk_id,
            document=doc
        )

    def vector_search(self, embedding: List[float], top_k: int = 5):
        query = {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": top_k
                }
            }
        }

        response = self.es.search(
            index=self.index_name,
            knn=query,
            source=["content", "source"]
        )
        return response.get("hits", {}).get("hits", [])