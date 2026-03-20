"""
Elasticsearch 벡터 스토어 모듈.

이 모듈은 RAG 시스템의 핵심인 하이브리드 검색 기능을 제공합니다.
- 벡터 검색(Semantic): 임베딩 기반 의미적 유사도 검색
- 키워드 검색(BM25): 전통적인 역색인 기반 텍스트 매칭
- 하이브리드 검색: 두 방식을 가중치로 결합하여 정확도 향상

핵심 설계 패턴:
- Double-Checked Locking: 멀티스레드 환경에서 안전한 싱글톤 초기화
- 비동기 I/O: aiohttp/AsyncElasticsearch로 논블로킹 처리
"""

import asyncio
import hashlib
import warnings
from datetime import datetime
from types import TracebackType
from typing import Any

import urllib3
from common.config import settings
from common.logger import logger
from elasticsearch import AsyncElasticsearch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.state import Document, RetrievedContext

# Cloudflared 터널 환경에서 SSL 인증서 검증 경고 억제
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", message=".*verify_certs.*")


class ElasticsearchStore:
    """
    비동기 Elasticsearch 벡터 스토어 관리 클래스.

    이 클래스는 문서의 저장, 검색, 삭제를 담당하며,
    벡터 유사도 검색과 키워드 검색을 모두 지원합니다.

    Attributes:
        _es_client: Elasticsearch 비동기 클라이언트 (지연 초기화)
        _embeddings: Ollama 임베딩 모델 인스턴스
        _text_splitter: 문서 청킹을 위한 텍스트 분할기
        _lock: 동시성 제어를 위한 비동기 락
        _initialized: 초기화 완료 플래그 (Double-Checked Locking용)
    """

    def __init__(self) -> None:
        self._es_client: AsyncElasticsearch | None = None
        self._embeddings: OllamaEmbeddings | None = None
        self._text_splitter: RecursiveCharacterTextSplitter | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def __aenter__(self) -> "ElasticsearchStore":
        """
        비동기 컨텍스트 매니저 진입 - 리소스 할당.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 생명주기 추적 (Telemetry)               │
        │ ──────────────────────────────────────────────────────────── │
        │ 진입 시점: async with ElasticsearchStore() as store:        │
        │ 할당 리소스:                                                 │
        │   - Elasticsearch AsyncClient 연결                          │
        │   - Ollama Embeddings 모델 인스턴스                         │
        │   - RecursiveCharacterTextSplitter 인스턴스                 │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug(
            "[ElasticsearchStore] __aenter__: 리소스 할당 시작 (id={})", id(self)
        )
        await self.initialize()
        logger.debug(
            "[ElasticsearchStore] __aenter__: 리소스 할당 완료 (ES 연결 활성화)"
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        비동기 컨텍스트 매니저 종료 - 리소스 해제.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 해제 보장 (메모리 누수 방지)            │
        │ ──────────────────────────────────────────────────────────── │
        │ 종료 시점: with 블록 종료 또는 예외 발생 시                  │
        │ 해제 리소스:                                                 │
        │   - Elasticsearch 연결 종료 (TCP 소켓 반환)                 │
        │   - 내부 상태 플래그 리셋                                    │
        │                                                             │
        │ 예외 발생 시에도 반드시 실행됨 (finally와 동일)              │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug(
            "[ElasticsearchStore] __aexit__: 리소스 해제 시작 (id={})", id(self)
        )
        if exc_type:
            logger.warning(
                "[ElasticsearchStore] __aexit__: 예외 감지 - {}: {}",
                exc_type.__name__,
                exc_val,
            )
        await self.close()
        logger.debug("[ElasticsearchStore] __aexit__: 리소스 해제 완료 (ES 연결 종료)")

    async def initialize(self) -> None:
        """
        벡터스토어 초기화 (Double-Checked Locking 패턴).

        ┌──────────────────────────────────────────────────────────────────────┐
        │              Double-Checked Locking 패턴 상세 설명                    │
        ├──────────────────────────────────────────────────────────────────────┤
        │ 문제 상황:                                                            │
        │   - 여러 코루틴이 동시에 initialize()를 호출할 수 있음                │
        │   - 락만 사용하면 이미 초기화된 후에도 매번 락 대기 발생 (성능 저하)  │
        │   - 락 없이 체크만 하면 동시 초기화로 리소스 낭비/충돌 발생           │
        │                                                                      │
        │ 해결책 (Double-Checked Locking):                                      │
        │                                                                      │
        │   코루틴 A          코루틴 B          코루틴 C                        │
        │      │                 │                 │                           │
        │      ▼                 ▼                 ▼                           │
        │   [1차 체크]       [1차 체크]       [1차 체크]                        │
        │   initialized?     initialized?     initialized?                     │
        │      │ No              │ No              │ No                        │
        │      ▼                 ▼                 ▼                           │
        │   [락 획득 시도]   [락 대기...]     [락 대기...]                      │
        │      │                                                               │
        │      ▼                                                               │
        │   [2차 체크] ← 중요! 다른 코루틴이 먼저 초기화했을 수 있음            │
        │   initialized?                                                       │
        │      │ No (아직 아무도 초기화 안 함)                                  │
        │      ▼                                                               │
        │   [실제 초기화 수행]                                                  │
        │   ES 클라이언트 생성                                                  │
        │   임베딩 모델 로드                                                    │
        │   initialized = True                                                 │
        │      │                                                               │
        │      ▼                                                               │
        │   [락 해제] ─────► 코루틴 B 락 획득                                   │
        │                       │                                              │
        │                       ▼                                              │
        │                   [2차 체크]                                         │
        │                   initialized? = True                                │
        │                       │                                              │
        │                       ▼                                              │
        │                   [즉시 반환] (초기화 스킵)                           │
        │                                                                      │
        │ 이후 모든 호출:                                                       │
        │   [1차 체크] initialized? = True → [즉시 반환] (락 없이 빠른 경로)    │
        └──────────────────────────────────────────────────────────────────────┘

        성능 이점:
        - 초기화 완료 후: O(1) 단순 플래그 체크만 수행, 락 오버헤드 없음
        - 초기화 중: 락으로 동시 초기화 방지, 2차 체크로 중복 초기화 방지
        """
        # ═══════════════════════════════════════════════════════════════
        # [1단계] 첫 번째 체크: 락 없이 빠른 경로
        # - 이미 초기화되었다면 락 획득 없이 즉시 반환
        # - 대부분의 호출(초기화 완료 후)은 여기서 바로 반환됨
        # ═══════════════════════════════════════════════════════════════
        if self._initialized:
            return

        async with self._lock:
            # ═══════════════════════════════════════════════════════════
            # [2단계] 두 번째 체크: 락 획득 후 재확인
            # - 락 대기 중에 다른 코루틴이 초기화를 완료했을 수 있음
            # - 이 체크가 없으면 동일한 초기화가 여러 번 실행될 수 있음
            # ═══════════════════════════════════════════════════════════
            if self._initialized:
                return

            ollama_headers: dict[str, str] = {}
            if settings.CF_ACCESS_CLIENT_ID and settings.CF_ACCESS_CLIENT_SECRET:
                ollama_headers = {
                    "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
                    "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
                }

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
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.EMBEDDING_MODEL,
                headers=ollama_headers,
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
                        "content": {"type": "text", "analyzer": "nori"},
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

    def _generate_doc_id(self, content: str, metadata: dict[str, Any]) -> str:
        """문서 고유 ID 생성"""
        content_hash = hashlib.md5(content.encode()).hexdigest()  # noqa: S324
        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()  # noqa: S324
        return f"{content_hash[:8]}_{metadata_hash[:8]}"

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """문서 추가"""
        await self.initialize()

        document_ids = []

        for doc in documents:
            chunks = await asyncio.to_thread(
                self._text_splitter.split_text, doc.content
            )

            # embed_documents: 문서용 인터페이스로 배치 처리 (embed_query는 쿼리용)
            # bge-m3 asymmetric retrieval 설계상 문서/쿼리 인코딩이 다름
            embeddings = await asyncio.to_thread(
                self._embeddings.embed_documents, chunks
            )

            origin_doc_id = doc.doc_id or self._generate_doc_id(
                doc.content, doc.metadata
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "origin_doc_id": origin_doc_id,
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
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
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
                "num_candidates": k * 10,
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
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievedContext:
        """키워드 기반 검색"""
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        search_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"content": {"query": query, "analyzer": "nori"}}}
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
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        vector_weight: float = 0.5,
    ) -> RetrievedContext:
        """
        하이브리드 검색: 벡터 유사도 + BM25 키워드 검색 결합.

        RRF(Reciprocal Rank Fusion) 방식으로 두 검색 결과를 병합:
        - 각 검색 결과의 순위(rank) 기반 점수 계산: 1 / (RRF_K + rank)
        - min-max 정규화 대비 스코어 outlier에 강건하고 안정적인 병합 보장
        - 두 검색 모두에서 등장한 문서는 RRF 점수가 누적되어 상위 랭크

        수식:
            rrf_score = vector_weight × (1/(60+vec_rank))
                      + (1-vector_weight) × (1/(60+kw_rank))

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 최대 문서 수
            filters: 메타데이터 기반 필터 조건
            vector_weight: 벡터 검색 RRF 가중치 (0.0~1.0, 기본값 0.5)

        Returns:
            RetrievedContext: RRF 점수로 정렬된 검색 결과
        """
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        # 두 검색을 병렬 실행하여 응답 시간 단축
        vector_results, keyword_results = await asyncio.gather(
            self.similarity_search(query, k * 2, filters),
            self.keyword_search(query, k * 2, filters),
        )

        # ═══════════════════════════════════════════════════════════════
        # RRF(Reciprocal Rank Fusion) 병합
        #
        # 기존 min-max 정규화의 문제:
        #   score / max_score → max_score가 outlier면 나머지 점수 왜곡
        #
        # RRF 해결책:
        #   순위(rank) 기반 점수 → 절대 점수 크기 무관, 상대 순서만 반영
        #   RRF_K=60: Cormack et al. 2009 논문 기본값, 상위/하위 rank 영향 균형
        # ═══════════════════════════════════════════════════════════════
        RRF_K = 60
        doc_scores: dict[str, tuple[Document, float]] = {}

        # [1단계] 벡터 검색 결과: rank → RRF 점수
        for rank, (doc, _) in enumerate(
            zip(vector_results.documents, vector_results.scores, strict=True), start=1
        ):
            doc_scores[doc.doc_id] = (doc, vector_weight / (RRF_K + rank))

        # [2단계] 키워드 검색 결과: rank → RRF 점수 누적
        for rank, (doc, _) in enumerate(
            zip(keyword_results.documents, keyword_results.scores, strict=True), start=1
        ):
            kw_rrf = (1 - vector_weight) / (RRF_K + rank)
            if doc.doc_id in doc_scores:
                existing_doc, existing_score = doc_scores[doc.doc_id]
                doc_scores[doc.doc_id] = (existing_doc, existing_score + kw_rrf)
            else:
                doc_scores[doc.doc_id] = (doc, kw_rrf)

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
