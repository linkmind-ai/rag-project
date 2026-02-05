import asyncio
import hashlib
import urllib3
import warnings
from typing import List, Optional, Dict, Any
from datetime import datetime
from elasticsearch import AsyncElasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.state import Document, RetrievedContext
from common.config import settings

# Cloudflared 환경: SSL 인증서 검증 비활성화 경고 억제
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message=".*verify_certs.*")


class ElasticsearchStore:
    """비동기 엘라스틱서치 벡터 스토어 관리 클래스"""

    def __init__(self):
        self._es_client: Optional[AsyncElasticsearch] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def __aenter__(self) -> "ElasticsearchStore":
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """비동기 컨텍스트 매니저 종료 - 리소스 정리"""
        await self.close()

    async def initialize(self) -> None:
        """벡터스토어 초기화"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if settings.ELASTICSEARCH_PASSWORD:
                self._es_client = AsyncElasticsearch(
                    [settings.ELASTICSEARCH_URL],
                    api_key=(
                        settings.ELASTICSEARCH_USER,
                        settings.ELASTICSEARCH_PASSWORD,
                    ),
                    verify_certs=False,
                )
            else:
                self._es_client = AsyncElasticsearch([settings.ELASTICSEARCH_URL])

            self._embeddings = OllamaEmbeddings(
                base_url=settings.OLLAMA_BASE_URL, model=settings.EMBEDDING_MODEL
            )

            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )

            await self._create_index()

            # await asyncio.to_thread(
            #     self._init_vectorstore
            # )

            self._initialized = True

    async def _create_index(self) -> None:
        """Elasticsearch 인덱스 생성"""
        index_exists = await self._es_client.indices.exists(
            index=settings.ELASTICSEARCH_INDEX
        )

        if not index_exists:
            index_mapping = {
                "mappings": {
                    "properties": {
                        "content": {"type": "text", "analyzer": "standard"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": settings.EMBEDDING_DIM,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "metadata": {"type": "object", "enabled": True},
                        "created_at": {"type": "date"},
                        "doc_hash": {"type": "keyword"},
                    }
                }
            }

            await self._es_client.indices.create(
                index=settings.ELASTICSEARCH_INDEX, body=index_mapping
            )

    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """문서 고유 ID 생성"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        return f"{content_hash[:8]}_{metadata_hash[:8]}"

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """문서 추가"""
        await self.initialize()

        document_ids = []

        for doc in documents:
            chunks = await asyncio.to_thread(
                self._text_splitter.split_text, doc.content
            )

            for i, chunk in enumerate(chunks):
                embedding = await asyncio.to_thread(self._embeddings.embed_query, chunk)

                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "origin_doc_id": doc.doc_id
                    or self._generate_doc_id(doc.content, doc.metadata),
                }
                doc_id = self._generate_doc_id(chunk, chunk_metadata)

                document_body = {
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": chunk_metadata,
                    "created_at": datetime.now().isoformat(),
                    "doc_hash": doc_id,
                }

                await self._es_client.index(
                    index=settings.ELASTICSEARCH_INDEX, id=doc_id, body=document_body
                )

                document_ids.append(doc_id)

        await self._es_client.indices.refresh(index=settings.ELASTICSEARCH_INDEX)

        return document_ids

    async def similarity_search(
        self, query: str, k: int = None, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievedContext:
        """유사도 검색"""
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        query_embedding = await asyncio.to_thread(self._embeddings.embed_query, query)

        search_query = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": k,
                "num_candidates": k * 2,
            },
            "_source": ["content", "metadata", "doc_hash"],
        }

        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {f"metadata.{key}": value}})

            search_query["query"] = {"bool": {"filter": filter_clauses}}

        response = await self._es_client.search(
            index=settings.ELASTICSEARCH_INDEX, body=search_query, size=k
        )

        documents = []
        scores = []

        for hit in response["hits"]["hits"]:
            documents.append(
                Document(
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    doc_id=hit["_source"].get("doc_hash", hit["_id"]),
                )
            )
            scores.append(float(hit["_score"]))

        total_hits = response["hits"]["total"]["value"]

        return RetrievedContext(
            documents=documents, scores=scores, total_hits=total_hits
        )

    async def keyword_search(
        self, query: str, k: int = None, filters: Optional[Dict[str, Any]] = None
    ) -> RetrievedContext:
        """키워드 기반 검색"""
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": {"query": query, "analyzer": "standard"}}}
                    ]
                }
            },
            "_source": ["content", "metadata", "doc_hash"],
        }

        if filters:
            filter_clauses = []
            for key, value in filters.items():
                filter_clauses.append({"term": {f"metadata.{key}": value}})

            search_query["query"]["bool"]["filter"] = filter_clauses

        response = await self._es_client.search(
            index=settings.ELASTICSEARCH_INDEX, body=search_query, size=k
        )

        documents = []
        scores = []

        for hit in response["hits"]["hits"]:
            documents.append(
                Document(
                    content=hit["_source"]["content"],
                    metadata=hit["_source"].get("metadata", {}),
                    doc_id=hit["_source"].get("doc_hash", hit["_id"]),
                )
            )
            scores.append(float(hit["_score"]))

        total_hits = response["hits"]["total"]["value"]

        return RetrievedContext(
            documents=documents, scores=scores, total_hits=total_hits
        )

    async def hybrid_search(
        self,
        query: str,
        k: int = None,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
    ) -> RetrievedContext:
        """유사도 + 키워드 기반 검색"""
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        vector_results, keyword_results = await asyncio.gather(
            self.similarity_search(query, k * 2, filters),
            self.keyword_search(query, k * 2, filters),
        )

        doc_scores: Dict[str, tuple[Document, float]] = {}

        max_vector_score = max(vector_results.scores) if vector_results.scores else 1.0
        for doc, score in zip(vector_results.documents, vector_results.scores):
            normalized_score = score / max_vector_score
            doc_scores[doc.doc_id] = (doc, normalized_score * vector_weight)

        max_keyword_score = (
            max(keyword_results.scores) if keyword_results.scores else 1.0
        )
        for doc, score in zip(keyword_results.documents, keyword_results.scores):
            normalized_score = score / max_keyword_score
            if doc.doc_id in doc_scores:
                existing_doc, existing_score = doc_scores[doc.doc_id]
                doc_scores[doc.doc_id] = (
                    existing_doc,
                    existing_score + normalized_score * (1 - vector_weight),
                )
            else:
                doc_scores[doc.doc_id] = (doc, normalized_score * (1 - vector_weight))

        sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)[
            :k
        ]

        documents = [doc for doc, _ in sorted_results]
        scores = [score for _, score in sorted_results]

        return RetrievedContext(
            documents=documents, scores=scores, total_hits=len(doc_scores)
        )

    async def delete_document(self, doc_id: str) -> bool:
        """문서 삭제"""
        await self.initialize()

        try:
            await self._es_client.delete(index=settings.ELASTICSEARCH_INDEX, id=doc_id)
            await self._es_client.indices.refresh(index=settings.ELASTICSEARCH_INDEX)
            return True
        except Exception:
            return False

    async def get_document_count(self) -> int:
        """문서 수량 조회"""
        await self.initialize()

        response = await self._es_client.count(index=settings.ELASTICSEARCH_INDEX)
        return response["count"]

    async def clear(self) -> None:
        """스토어 초기화"""
        await self.initialize()

        async with self._lock:
            await self._es_client.indices.delete(index=settings.ELASTICSEARCH_INDEX)
            await self._create_index()

    async def health_check(self) -> bool:
        """엘라스틱서치 연결상태 확인"""
        try:
            if not self._es_client:
                return False
            return await self._es_client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """연결 종료"""
        if self._es_client:
            await self._es_client.close()


elasticsearch_store = ElasticsearchStore()
