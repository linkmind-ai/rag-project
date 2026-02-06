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
from typing import Any

import urllib3
from common.config import settings
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
        print(f"[ElasticsearchStore] __aenter__: 리소스 할당 시작 (id={id(self)})")
        await self.initialize()
        print("[ElasticsearchStore] __aenter__: 리소스 할당 완료 (ES 연결 활성화)")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
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
        print(f"[ElasticsearchStore] __aexit__: 리소스 해제 시작 (id={id(self)})")
        if exc_type:
            print(
                f"[ElasticsearchStore] __aexit__: 예외 감지 - "
                f"{exc_type.__name__}: {exc_val}"
            )
        await self.close()
        print("[ElasticsearchStore] __aexit__: 리소스 해제 완료 (ES 연결 종료)")

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

    def _generate_doc_id(self, content: str, metadata: dict[str, Any]) -> str:
        """문서 고유 ID 생성"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        metadata_str = str(sorted(metadata.items()))
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        return f"{content_hash[:8]}_{metadata_hash[:8]}"

    async def add_documents(self, documents: list[Document]) -> list[str]:
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
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        vector_weight: float = 0.5,
    ) -> RetrievedContext:
        """
        하이브리드 검색: 벡터 유사도 + BM25 키워드 검색 결합.

        두 검색 방식의 장점을 결합하여 검색 품질을 향상시킵니다:
        - 벡터 검색: 의미적 유사성 포착 (동의어, 유사 개념)
        - 키워드 검색: 정확한 용어 매칭 (고유명사, 전문용어)

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 최대 문서 수
            filters: 메타데이터 기반 필터 조건
            vector_weight: 벡터 검색 가중치 (0.0~1.0, 기본값 0.5)
                          키워드 가중치는 자동으로 (1 - vector_weight)

        Returns:
            RetrievedContext: 정규화된 점수로 정렬된 검색 결과
        """
        await self.initialize()

        k = k or settings.TOP_K_RESULTS

        # 두 검색을 병렬 실행하여 응답 시간 단축
        # k*2개씩 가져와서 병합 후 상위 k개 선택
        vector_results, keyword_results = await asyncio.gather(
            self.similarity_search(query, k * 2, filters),
            self.keyword_search(query, k * 2, filters),
        )

        doc_scores: dict[str, tuple[Document, float]] = {}

        # ┌──────────────────────────────────────────────────────────────────────┐
        # │                하이브리드 스코어 정규화 수식 상세 설명                 │
        # ├──────────────────────────────────────────────────────────────────────┤
        # │ 문제: 두 검색 방식의 스코어 범위가 다름                               │
        # │                                                                      │
        # │   벡터 검색 (코사인 유사도)          BM25 키워드 검색                 │
        # │   ┌────────────────────┐           ┌────────────────────┐           │
        # │   │ 범위: 0.0 ~ 1.0    │           │ 범위: 0.0 ~ ∞      │           │
        # │   │ 예: 0.85, 0.72     │           │ 예: 12.5, 8.3      │           │
        # │   └────────────────────┘           └────────────────────┘           │
        # │                                                                      │
        # │   → 단순 합산 시 BM25 스코어가 결과를 지배하게 됨                     │
        # │                                                                      │
        # │ 해결: Min-Max 정규화로 0~1 범위 통일 후 가중치 적용                   │
        # │                                                                      │
        # │   정규화 공식:                                                        │
        # │   ┌─────────────────────────────────────┐                            │
        # │   │  normalized_score = score / max_score                            │
        # │   └─────────────────────────────────────┘                            │
        # │                                                                      │
        # │   예시 (vector_weight = 0.5):                                        │
        # │   ┌───────────────────────────────────────────────────────────────┐  │
        # │   │ 문서 A:                                                       │  │
        # │   │   벡터 스코어: 0.85 → 정규화: 0.85/0.85 = 1.0 → 가중: 0.5    │  │
        # │   │   키워드 스코어: 12.5 → 정규화: 12.5/12.5 = 1.0 → 가중: 0.5  │  │
        # │   │   최종 스코어: 0.5 + 0.5 = 1.0                                │  │
        # │   │                                                               │  │
        # │   │ 문서 B (벡터에서만 발견):                                      │  │
        # │   │   벡터 스코어: 0.72 → 정규화: 0.72/0.85 = 0.847 → 가중: 0.424│  │
        # │   │   최종 스코어: 0.424                                          │  │
        # │   │                                                               │  │
        # │   │ 문서 C (키워드에서만 발견):                                    │  │
        # │   │   키워드 스코어: 8.3 → 정규화: 8.3/12.5 = 0.664 → 가중: 0.332│  │
        # │   │   최종 스코어: 0.332                                          │  │
        # │   └───────────────────────────────────────────────────────────────┘  │
        # │                                                                      │
        # │   최종 수식:                                                          │
        # │   ┌─────────────────────────────────────────────────────────────────┐│
        # │   │ final_score = (vec_score/max_vec × W) + (kw_score/max_kw × 1-W) ││
        # │   │                                                                 ││
        # │   │ 여기서 W = vector_weight (기본값 0.5)                           ││
        # │   └─────────────────────────────────────────────────────────────────┘│
        # └──────────────────────────────────────────────────────────────────────┘

        # ═══════════════════════════════════════════════════════════════
        # [1단계] 벡터 검색 결과 정규화 및 가중치 적용
        # ═══════════════════════════════════════════════════════════════
        max_vector_score = max(vector_results.scores) if vector_results.scores else 1.0
        for doc, score in zip(
            vector_results.documents, vector_results.scores, strict=True
        ):
            # 정규화: 현재 스코어 / 최대 스코어 (0~1 범위로 변환)
            normalized_score = score / max_vector_score
            # 가중치 적용: 정규화 스코어 × 벡터 가중치
            doc_scores[doc.doc_id] = (doc, normalized_score * vector_weight)

        # ═══════════════════════════════════════════════════════════════
        # [2단계] 키워드 검색 결과 정규화 및 가중치 적용
        # ═══════════════════════════════════════════════════════════════
        max_keyword_score = (
            max(keyword_results.scores) if keyword_results.scores else 1.0
        )
        for doc, score in zip(
            keyword_results.documents, keyword_results.scores, strict=True
        ):
            normalized_score = score / max_keyword_score
            if doc.doc_id in doc_scores:
                # ┌────────────────────────────────────────────────────────┐
                # │ 두 검색 모두에서 발견된 문서: 점수 합산                  │
                # │ → 의미적으로도 유사하고 키워드도 매칭되는 "황금 문서"   │
                # │ → Reciprocal Rank Fusion과 유사한 효과                  │
                # └────────────────────────────────────────────────────────┘
                existing_doc, existing_score = doc_scores[doc.doc_id]
                doc_scores[doc.doc_id] = (
                    existing_doc,
                    existing_score + normalized_score * (1 - vector_weight),
                )
            else:
                # 키워드 검색에서만 발견된 문서 (정확한 용어 매칭)
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
