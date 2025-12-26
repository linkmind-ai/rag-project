from sentence_transformers import SentenceTransformer
import numpy as np

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from elasticsearch import Elasticsearch
import numpy as np


class HybridSearcher:
    def __init__(self, es_host, es_id, es_api_key, index_name, embedding_model_name, reranker_model_name):
        self.es = Elasticsearch(es_host,
                                api_key = (es_id, es_api_key))
        self.index_name = index_name
        self.embedder = SentenceTransformer(embedding_model_name)
        self.reranker = FlagReranker(reranker_model_name, use_fp16=True)


    def encode_query(self, query: str):
        """Query → float32 vector 변환"""
        vec = self.embedder.encode(query)
        return np.asarray(vec, dtype=np.float32).tolist()


    def search(self, query: str, k=3, bm25_weight=0.5):
        """
        Hybrid search:
        - BM25 + Vector cosine similarity (ES 내부에서 가중합)
        - BGE reranker로 최종 re-ranking

        Parameters:
            query: 검색문
            k: 최종 반환 문서 수
            topN: reranker를 위해 ES에서 먼저 뽑을 문서 수
            bm25_weight: BM25 가중치 (0~1)

        Returns:
            리스트: reranker가 재정렬한 top-k 문서
        """

        query_vector = self.encode_query(query)

        # === Elasticsearch Hybrid Search ===
        es_res = self.es.search(
            index=self.index_name,
            query={
                "script_score": {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content", "metadata.title", "metadata.keywords"]
                        }
                    },
                    "script": {
                        "source": """
                            double bm25 = _score;
                            double cos = cosineSimilarity(params.query_vector, 'embedding');

                            // cosine은 -1~1 범위이므로 0~1 범위로 normalize
                            double cos_norm = (cos + 1) / 2;

                            double bm25_weight = params.bm25_weight;
                            double hybrid = bm25_weight * bm25 + (1 - bm25_weight) * cos_norm;

                            return hybrid;
                        """,
                        "params": {
                            "query_vector": query_vector,
                            "bm25_weight": bm25_weight
                        }
                    }
                }
            },
            size=k*3,
            _source_excludes=["embedding"]
        )

        # 후보군 문서 추출
        hits = es_res["hits"]["hits"]
        candidate_docs = [hit["_source"]["content"] for hit in hits]

        # === BGE reranker ===
        pairs = [[query, doc] for doc in candidate_docs]
        scores = self.reranker.compute_score(pairs)

        # re-ranking
        reranked = [
            doc for _, doc in sorted(
                zip(scores, candidate_docs),
                key=lambda x: x[0],
                reverse=True
            )
        ]

        return reranked[:k]