"""
LangGraph 기반 RAG 워크플로우 모듈.

이 모듈은 검색-증강-생성(RAG) 파이프라인의 핵심 오케스트레이션을 담당합니다.

워크플로우 구조 (3-노드 순차 실행):
┌─────────────┐    ┌─────────────┐    ┌───────────────────┐
│  retrieve   │ -> │  generate   │ -> │ identify_evidence │
│ (하이브리드 │    │ (LLM 응답   │    │ (근거 문서 식별)  │
│  검색)      │    │  생성)      │    │                   │
└─────────────┘    └─────────────┘    └───────────────────┘

핵심 기능:
- N1: 하이브리드 검색 (벡터 + BM25)
- N2: 컨텍스트 기반 응답 생성 (대화 이력 지원)
- N3: LLM + 키워드 하이브리드 근거 식별
"""

import os
import asyncio
import json
import re
from types import TracebackType
from typing import Any
import uuid

from common.config import settings
from common.logger import logger
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from models.state import Document, GraphState, Message
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT
from prompts.grade_document_prompt import GradeDocuments, _GRADE_PROMPT
from prompts.query_rewrite_for_web_prompt import _REWRITE_FOR_WEB_SEARCH_PROMPT
from stores.vector_store import elasticsearch_store

# ========== 한국어 불용어 (Stopwords) ==========
# 키워드 기반 근거 식별 시 노이즈를 제거하기 위한 불용어 집합.
# 조사, 접속사, 대명사 등 의미적 가치가 낮은 단어들을 필터링하여
# 핵심 키워드 매칭의 정확도를 높입니다.
# ==============================================
_KOREAN_STOPWORDS: set[str] = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "와",
    "과",
    "도",
    "로",
    "으로",
    "만",
    "까지",
    "부터",
    "처럼",
    "같이",
    "보다",
    "라고",
    "하고",
    "그",
    "저",
    "이것",
    "그것",
    "저것",
    "여기",
    "거기",
    "저기",
    "무엇",
    "어디",
    "있다",
    "없다",
    "하다",
    "되다",
    "이다",
    "아니다",
    "그리고",
    "그러나",
    "하지만",
    "또는",
    "혹은",
    "및",
    "등",
    "때문",
    "위해",
    "통해",
    "대해",
    "관해",
}


class RAGGraph:
    """
    LangGraph 기반 RAG 워크플로우 관리 클래스.

    이 클래스는 검색-생성-근거식별의 3단계 파이프라인을 구성하고 실행합니다.
    StateGraph를 사용하여 각 노드 간 상태 전달과 순차 실행을 보장합니다.

    Attributes:
        _llm: Ollama LLM 인스턴스 (지연 초기화)
        _graph: 컴파일된 LangGraph StateGraph
        _initialized: 초기화 완료 플래그 (Double-Checked Locking용)
        _lock: 동시성 제어를 위한 비동기 락
        chat_prompt: 단일 쿼리용 프롬프트 템플릿
        chat_with_history_prompt: 대화 이력 포함 프롬프트 템플릿
        get_evidence_prompt: 근거 문서 식별용 프롬프트 템플릿
    """

    def __init__(self) -> None:
        self._llm: Ollama | None = None
        self._graph = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT
        self.get_evidence_prompt = _GET_EVIDENCE_PROMPT
        self.grade_document_prompt = _GRADE_PROMPT
        self.query_rewrite_prompt = _REWRITE_FOR_WEB_SEARCH_PROMPT

    async def __aenter__(self) -> "RAGGraph":
        """
        비동기 컨텍스트 매니저 진입 - LLM 및 그래프 초기화.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 생명주기 추적 (Telemetry)               │
        │ ──────────────────────────────────────────────────────────── │
        │ 진입 시점: async with CRAGGraph() as graph:                  │
        │ 할당 리소스:                                                 │
        │   - Ollama LLM 인스턴스 (HTTP 연결 포함)                    │
        │   - LangGraph StateGraph 컴파일된 워크플로우                │
        │   - 프롬프트 템플릿 바인딩                                   │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug(
            "[CRAGGraph] __aenter__: LLM 및 그래프 초기화 시작 (id={})", id(self)
        )
        await self.initialize()
        logger.info(
            "[CRAGGraph] __aenter__: 초기화 완료 (모델: {})", settings.OLLAMA_MODEL
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        비동기 컨텍스트 매니저 종료 - 연결 리소스 정리.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 해제 보장 (메모리 누수 방지)            │
        │ ──────────────────────────────────────────────────────────── │
        │ 종료 시점: with 블록 종료 또는 예외 발생 시                  │
        │ 해제 리소스:                                                 │
        │   - ElasticsearchStore 연결 종료 (위임)                     │
        │   - 초기화 플래그 리셋                                       │
        │                                                             │
        │ 참고: LLM 인스턴스는 stateless하여 별도 종료 불필요         │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug("[CRAGGraph] __aexit__: 리소스 정리 시작 (id={})", id(self))
        if exc_type:
            logger.warning(
                "[CRAGGraph] __aexit__: 예외 감지 - {}: {}", exc_type.__name__, exc_val
            )
        await self.close()
        logger.debug("[CRAGGraph] __aexit__: 리소스 정리 완료")

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                headers={
                    "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
                    "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
                },
                temperature=1.0,
            )

            parser = PydanticOutputParser(pydantic_object=GradeDocuments)
            self._document_grader = self.grade_document_prompt | self._llm | parser
            self._query_rewriter = (
                self.query_rewrite_prompt | self._llm | StrOutputParser()
            )

            os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
            self._web_search_tool = TavilySearchResults(max_results=2)
            # self._prompt_compressor = None

            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        """
        LangGraph 워크플로우 구조 빌드.

        ┌──────────────────────────────────────────────────────────────────────┐
        │                    LangGraph 상태 전이 과정 상세                       │
        ├──────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  [입력]                                                               │
        │  GraphState {                                                        │
        │    query: "사용자 질문",                                             │
        │    session_id: "세션ID",                                             │
        │    chat_history: [이전 대화],                                        │
        │    retrieved_docs: [],      ← 비어있음                               │
        │    answer: "",              ← 비어있음                               │
        │    evidence_indices: []     ← 비어있음                               │
        │  }                                                                   │
        │      │                                                               │
        │      ▼                                                               │
        │  ┌─────────────────────────────────────────────────────────────┐     │
        │  │ [N1] retrieve 노드                                          │     │
        │  │ ─────────────────────────────────────────────────────────── │     │
        │  │ 입력: query                                                 │     │
        │  │ 처리: elasticsearch_store.hybrid_search() 호출              │     │
        │  │ 출력: {"retrieved_docs": [문서1, 문서2, 문서3]}             │     │
        │  │       ↓ 상태 병합 (기존 상태 + 출력)                        │     │
        │  └─────────────────────────────────────────────────────────────┘     │
        │      │                                                               │
        │      │ edge: "retrieve" → "generate"                                │
        │      ▼                                                               │
        │  ┌─────────────────────────────────────────────────────────────┐     │
        │  │ [N2] generate 노드                                          │     │
        │  │ ─────────────────────────────────────────────────────────── │     │
        │  │ 입력: query, retrieved_docs, chat_history                   │     │
        │  │ 처리: LLM에 프롬프트 + 컨텍스트 전달                        │     │
        │  │ 출력: {"answer": "LLM이 생성한 답변"}                        │     │
        │  │       ↓ 상태 병합                                           │     │
        │  └─────────────────────────────────────────────────────────────┘     │
        │      │                                                               │
        │      │ edge: "generate" → "identify_evidence"                       │
        │      ▼                                                               │
        │  ┌─────────────────────────────────────────────────────────────┐     │
        │  │ [N3] identify_evidence 노드                                 │     │
        │  │ ─────────────────────────────────────────────────────────── │     │
        │  │ 입력: query, answer, retrieved_docs                         │     │
        │  │ 처리: 하이브리드 근거 식별 (LLM + 키워드 매칭)              │     │
        │  │ 출력: {"evidence_indices": [0, 2]}                          │     │
        │  │       ↓ 상태 병합                                           │     │
        │  └─────────────────────────────────────────────────────────────┘     │
        │      │                                                               │
        │      │ edge: "identify_evidence" → END                              │
        │      ▼                                                               │
        │  [최종 출력]                                                          │
        │  GraphState {                                                        │
        │    query: "사용자 질문",                                             │
        │    session_id: "세션ID",                                             │
        │    chat_history: [이전 대화],                                        │
        │    retrieved_docs: [문서1, 문서2, 문서3],  ← N1에서 채워짐           │
        │    answer: "LLM 답변",                     ← N2에서 채워짐           │
        │    evidence_indices: [0, 2]               ← N3에서 채워짐           │
        │  }                                                                   │
        │                                                                      │
        │  ※ 상태 병합 규칙: 노드 반환값의 키가 기존 상태를 덮어씀             │
        │  ※ 반환하지 않은 키는 이전 값 유지                                   │
        └──────────────────────────────────────────────────────────────────────┘
        """
        workflow = StateGraph(GraphState)

        # 노드 등록: 각 노드는 상태를 받아 부분 상태를 반환
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("query_rewrite", self._query_rewrite_node)
        workflow.add_node("search_web", self._web_search_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("identify_evidence", self._identify_evidence_node)

        # 진입점 설정: 그래프 실행 시 첫 번째로 실행될 노드
        workflow.set_entry_point("retrieve")

        # 엣지 연결: 노드 간 순차 실행 순서 정의
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_web_search,
            {
                "query_rewrite": "query_rewrite",
                "generate": "generate",
            },
        )
        workflow.add_edge("query_rewrite", "search_web")
        workflow.add_edge("search_web", "generate")

        workflow.add_edge("generate", "identify_evidence")
        workflow.add_edge("identify_evidence", END)  # END는 그래프 종료 마커

        # 컴파일: 정의된 그래프를 실행 가능한 형태로 변환
        self._graph = workflow.compile()

    async def _retrieve_node(self, state: GraphState) -> dict[str, Any]:
        """검색 노드"""
        query = state.query

        context = await elasticsearch_store.hybrid_search(
            query=query, k=settings.TOP_K_RESULTS, vector_weight=0.5
        )

        return {"retrieved_docs": context.documents}

    async def _grade_documents_node(self, state: GraphState):
        """검색된 chunk가 query와 관련이 있는지를 yes 또는 no로 체크"""
        query = state.query
        retrieved_docs = state.retrieved_docs

        # 필터링된 문서
        filtered_docs = []

        for doc in retrieved_docs:
            # Question - Document 의 관련성 평가
            input_data = {"document": doc.content, "query": query}
            score = await asyncio.to_thread(self._document_grader.invoke, input_data)

            if score.binary_score == "yes":
                filtered_docs.append(doc)

        # 관련 문서가 없으면 웹 검색 수행
        web_search = len(filtered_docs) == 0
        return {"retrieved_docs": filtered_docs, "web_search": web_search}

    async def _query_rewrite_node(self, state: GraphState) -> dict[str, Any]:
        """기존의 쿼리를 웹 검색에 최적화된 쿼리로 재작성"""
        query = state.query

        # 웹 검색을 위한 질문으로 재작성
        input_data = {"query": query}
        query_for_web_search = await asyncio.to_thread(
            self._query_rewriter.invoke, input_data
        )
        return {"query_for_web_search": query_for_web_search}

    async def _web_search_node(self, state: GraphState) -> dict[str, Any]:
        """refined_question으로 웹 검색"""
        query_for_web_search = state.query_for_web_search
        retrieved_docs = state.retrieved_docs

        # 웹 검색
        docs = await asyncio.to_thread(
            self._web_search_tool.invoke, query_for_web_search
        )

        # 검색 결과를 문서 형식으로 변환
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(
            doc_id=f"web_{uuid.uuid4()}",
            content=web_results,
            metadata={"source": "web"},
        )
        new_docs = retrieved_docs + [web_results]

        return {"retrieved_docs": new_docs}

    # async def _prompt_compression_node(self, state: GraphState) -> dict[str, Any]:
    #     """query와 documents를 결합해서 프롬프트 생성 후 LongLLMLingua로 압축"""
    #     query = state.query
    #     retrieved_docs = state.retrieved_docs

    #     if self._prompt_compressor is None:
    #         self._prompt_compressor = PromptCompressor("microsoft/phi-2")

    #     context_text = self._build_context(retrieved_docs)

    #     results = self._prompt_compressor.compress_prompt(
    #         context_text,
    #         question=query,
    #         ratio=0.55,
    #         # Set the special parameter for LongLLMLingua
    #         condition_in_question="after_condition",
    #         reorder_context="sort",
    #         dynamic_context_compression_ratio=0.3,
    #         condition_compare=True,
    #         context_budget="+100",
    #         rank_method="longllmlingua",
    #     )

    #     compressed_prompt = results["compressed_prompt"]

    #     return {"prompt": compressed_prompt}

    def _decide_to_web_search(self, state: GraphState):
        """grade_documents 노드에서 판별한 웹 검색 필요여부에 따라 쿼리를 routing"""
        web_search = state.web_search

        if web_search:
            # 웹 검색으로 정보 보강이 필요한 경우
            print("==== [DECISION: QUERY REWRITE FOR WEB SEARCH] ====")
            # 쿼리 재작성 노드로 라우팅
            return "query_rewrite"
        else:
            # 관련 문서가 존재하므로 답변 생성 단계(generate) 로 진행
            print("==== [DECISION: GENERATE] ====")
            return "generate"

    def _prepare_messages(self, chat_history: list[Message]) -> list[Any]:
        """
        대화 이력을 LangChain 메시지 형식으로 변환.

        Args:
            chat_history: Message 객체 리스트 (role, content 포함)

        Returns:
            LangChain 메시지 객체 리스트 (HumanMessage, AIMessage, SystemMessage)
        """
        messages = []
        for msg in chat_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            else:
                messages.append(SystemMessage(content=msg.content))
        return messages  # 버그 수정: return 문 누락되어 있었음

    def _build_context(self, retrieved_docs: list[Document]) -> str:
        """검색 문서 텍스트 변환"""
        context_text = "\n\n".join(
            [
                f"Document {i + 1}:\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        return context_text

    async def _generate_node(self, state: GraphState) -> dict[str, Any]:
        """응답 생성 노드"""
        query = state.query
        chat_history = state.chat_history
        retrieved_docs = state.retrieved_docs

        context_text = self._build_context(retrieved_docs)

        if chat_history:
            messages = self._prepare_messages(chat_history)
            prompt = self.chat_with_history_prompt
            input_data = {"context": context_text, "history": messages, "query": query}

        else:
            prompt = self.chat_prompt
            input_data = {"context": context_text, "query": query}

        chain = prompt | self._llm

        answer = await asyncio.to_thread(chain.invoke, input_data)

        return {"answer": answer}

    def _extract_keywords(self, text: str, min_length: int = 2) -> set[str]:
        """텍스트에서 핵심 키워드 추출 (불용어 제거)"""
        # 한글, 영문, 숫자만 추출
        tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
        keywords = {
            token.lower()
            for token in tokens
            if len(token) >= min_length and token not in _KOREAN_STOPWORDS
        }
        return keywords

    def _calculate_keyword_overlap(
        self, answer: str, doc_content: str
    ) -> tuple[float, set[str]]:
        """답변과 문서 간 키워드 일치도 계산"""
        answer_keywords = self._extract_keywords(answer)
        doc_keywords = self._extract_keywords(doc_content)

        if not answer_keywords:
            return 0.0, set()

        matched = answer_keywords & doc_keywords
        overlap_ratio = len(matched) / len(answer_keywords)

        return overlap_ratio, matched

    def _get_keyword_based_evidence(
        self,
        answer: str,
        retrieved_docs: list[Document],
        threshold: float = 0.1,
    ) -> list[tuple[int, float, set[str]]]:
        """키워드 기반 근거 문서 후보 추출"""
        candidates = []
        for idx, doc in enumerate(retrieved_docs):
            overlap_ratio, matched_keywords = self._calculate_keyword_overlap(
                answer, doc.content
            )
            if overlap_ratio >= threshold:
                candidates.append((idx, overlap_ratio, matched_keywords))

        # 일치도 높은 순으로 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    async def _parse_evidence_response(self, response: str) -> list[int]:
        """LLM 응답 근거 인덱스 파싱"""
        try:
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r"\{.*?\}", response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response

            parsed = json.loads(json_str)

            if isinstance(parsed, dict) and "evidence_indices" in parsed:
                indices = parsed["evidence_indices"]
                if isinstance(indices, list):
                    return [int(idx) for idx in indices]

            if isinstance(parsed, list):
                return [int(idx) for idx in parsed]

            return []

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("근거 응답 파싱 오류: {} | 원본: {}", e, response)

            numbers = re.findall(r"\d+", response)
            if numbers:
                return [int(n) for n in numbers]

            return []

    async def _identify_evidence_node(self, state: GraphState) -> dict[str, Any]:
        """
        답변 근거 문서 식별 노드 (N3 하이브리드 로직).

        이 노드는 LLM 기반 추론과 키워드 매칭을 결합하여
        답변의 근거가 된 문서를 정확하게 식별합니다.

        ┌─────────────────────────────────────────────────────────┐
        │                N3 하이브리드 근거 식별 로직              │
        ├─────────────────────────────────────────────────────────┤
        │ 1단계: 키워드 기반 후보 추출                             │
        │   - 답변과 문서 간 키워드 일치도 계산                    │
        │   - 불용어 제거 후 핵심 키워드만 비교                    │
        │   - 임계값(10%) 이상인 문서를 후보로 선정                │
        │                                                         │
        │ 2단계: LLM 기반 근거 식별                                │
        │   - 프롬프트로 LLM에게 근거 문서 인덱스 요청             │
        │   - JSON 파싱으로 인덱스 추출                            │
        │                                                         │
        │ 3단계: 하이브리드 결합                                   │
        │   - LLM 결과를 기본으로 사용                             │
        │   - 키워드 일치도 20% 이상 문서 보강 (LLM이 놓친 경우)   │
        │   - 키워드 일치도 5% 미만 문서 제거 (LLM 환각 방지)      │
        └─────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────┐
        │            Adaptive Thresholding (동적 임계값 적응) 설계            │
        ├─────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  "왜 하필 5%, 10%, 20%인가?"에 대한 설계 근거:                       │
        │                                                                     │
        │  현재 고정 임계값 (Baseline):                                        │
        │  ┌───────────┬────────────────────────────────────────────────────┐ │
        │  │ 5% 미만   │ LLM 환각 의심 → 제거                               │ │
        │  │ 10% 이상  │ 키워드 후보 진입 → 1단계 필터                       │ │
        │  │ 20% 이상  │ 고신뢰 문서 → LLM 결과 보강                        │ │
        │  └───────────┴────────────────────────────────────────────────────┘ │
        │                                                                     │
        │  동적 적응 확장 방안 (Future Enhancement):                           │
        │  ─────────────────────────────────────────────────────────────────  │
        │  1. 답변 길이 기반 조정:                                             │
        │     - 짧은 답변 (<100자): 키워드 수가 적으므로 임계값 ↓ (3%, 7%, 15%)│
        │     - 긴 답변 (>500자): 키워드 풍부하므로 임계값 ↑ (7%, 15%, 25%)   │
        │                                                                     │
        │     예시 수식:                                                       │
        │     base_threshold = 0.10                                           │
        │     length_factor = min(1.5, max(0.5, len(answer) / 300))          │
        │     adaptive_threshold = base_threshold * length_factor            │
        │                                                                     │
        │  2. 검색 스코어 분포 기반 조정:                                       │
        │     - 문서 간 스코어 분산이 큰 경우: 명확한 구분 → 임계값 ↑          │
        │     - 문서 간 스코어 분산이 작은 경우: 모호한 경계 → 임계값 ↓        │
        │                                                                     │
        │  3. 도메인별 튜닝:                                                   │
        │     - 전문 용어 밀도가 높은 도메인: 정확 매칭 중요 → 임계값 ↑        │
        │     - 일상어 기반 도메인: 의미적 매칭 중요 → 임계값 ↓               │
        │                                                                     │
        │  ※ 현재는 범용 Baseline으로 고정값 사용                              │
        │  ※ 실 데이터 분석 후 적응형 로직으로 확장 가능                       │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            state: 현재 그래프 상태 (query, answer, retrieved_docs 포함)

        Returns:
            Dict with 'evidence_indices': 근거 문서 인덱스 리스트
        """
        query = state.query
        answer = state.answer
        retrieved_docs = state.retrieved_docs

        if not retrieved_docs:
            return {"evidence_indices": []}

        # ═══════════════════════════════════════════════════════════════════
        # [1단계] 키워드 기반 후보 추출
        # 임계값 10%: 답변 키워드의 10% 이상이 문서에서 발견되면 후보로 선정
        # → 동적 적응 시 답변 길이에 따라 7%~15% 범위로 조정 가능
        # ═══════════════════════════════════════════════════════════════════
        keyword_candidates = self._get_keyword_based_evidence(
            answer, retrieved_docs, threshold=0.1
        )
        keyword_indices = {idx for idx, _, _ in keyword_candidates}

        logger.debug("[N3] 키워드 기반 후보: {}", keyword_candidates[:5])

        # ========== 2단계: LLM 기반 근거 식별 ==========
        documents_text = "\n\n".join(
            [
                f"[인덱스 {i}]\n"
                f"제목: {doc.metadata.get('title', '제목 없음')}\n"
                f"내용: {doc.content[:500]}"
                for i, doc in enumerate(retrieved_docs)
            ]
        )

        prompt = self.get_evidence_prompt
        chain = prompt | self._llm

        input_data = {
            "query": query,
            "answer": answer,
            "documents": documents_text,
        }

        try:
            result = await asyncio.to_thread(chain.invoke, input_data)
            llm_indices = await self._parse_evidence_response(result)

            # 유효 범위 필터링 (인덱스 범위 초과 방지)
            llm_indices = [idx for idx in llm_indices if 0 <= idx < len(retrieved_docs)]
            logger.debug("[N3] LLM 선택 인덱스: {}", llm_indices)

            # ═══════════════════════════════════════════════════════════════
            # [3단계] 하이브리드 결합: LLM + 키워드 매칭 교차 검증
            # ═══════════════════════════════════════════════════════════════
            final_indices_set = set(llm_indices)

            # ┌─────────────────────────────────────────────────────────────┐
            # │ 3-1. 보강 임계값 (REINFORCE_THRESHOLD = 20%)                │
            # │ ─────────────────────────────────────────────────────────── │
            # │ 목적: LLM이 놓친 고신뢰 문서 복구                            │
            # │ 근거: 답변 키워드의 20% 이상이 문서에 존재하면               │
            # │       실제로 참조된 것으로 볼 수 있음                        │
            # │                                                             │
            # │ 동적 적응 시:                                                │
            # │ - 짧은 답변: 15%로 낮춤 (키워드 수가 적어 매칭 어려움)       │
            # │ - 긴 답변: 25%로 높임 (키워드 풍부해 엄격 기준 적용)         │
            # └─────────────────────────────────────────────────────────────┘
            reinforce_threshold = 0.2  # 보강 임계값 (현재 고정, 추후 동적 조정 가능)
            for idx, overlap_ratio, matched in keyword_candidates:
                if overlap_ratio >= reinforce_threshold:
                    if idx not in final_indices_set:
                        logger.debug(
                            "[N3] 키워드 기반 보강: 문서 {} (일치도: {:.2%}, 키워드: {})",
                            idx,
                            overlap_ratio,
                            list(matched)[:5],
                        )
                    final_indices_set.add(idx)

            # ┌─────────────────────────────────────────────────────────────┐
            # │ 3-2. 제거 임계값 (HALLUCINATION_THRESHOLD = 5%)             │
            # │ ─────────────────────────────────────────────────────────── │
            # │ 목적: LLM 환각으로 잘못 선택된 문서 필터링                   │
            # │ 근거: 키워드 일치도가 5% 미만이면 답변 생성에                │
            # │       실제로 기여하지 않았을 가능성 높음                     │
            # │                                                             │
            # │ 동적 적응 시:                                                │
            # │ - 전문 용어 도메인: 3%로 낮춤 (정확 매칭 어려움)             │
            # │ - 일반 도메인: 7%로 높임 (일상어는 매칭 쉬움)                │
            # │                                                             │
            # │ 주의: 너무 높이면 정당한 근거 문서도 제거될 수 있음          │
            # └─────────────────────────────────────────────────────────────┘
            hallucination_threshold = 0.05  # 환각 제거 임계값
            for idx in list(final_indices_set):
                if idx not in keyword_indices:
                    overlap, _ = self._calculate_keyword_overlap(
                        answer, retrieved_docs[idx].content
                    )
                    if overlap < hallucination_threshold:
                        logger.debug(
                            "[N3] 낮은 일치도로 제거: 문서 {} (일치도: {:.2%})",
                            idx,
                            overlap,
                        )
                        final_indices_set.discard(idx)

            final_indices = sorted(final_indices_set)
            logger.info("[N3] 최종 evidence_indices: {}", final_indices)

            return {"evidence_indices": final_indices}

        except Exception as e:
            logger.error("근거 문서 식별 오류: {}", e)
            # Fallback: LLM 실패 시 키워드 기반 결과만 사용
            fallback_indices = [
                idx for idx, ratio, _ in keyword_candidates if ratio >= 0.15
            ]
            return {"evidence_indices": fallback_indices if fallback_indices else [0]}

    def get_graph(self) -> CompiledStateGraph | None:
        """컴파일된 그래프 반환"""
        return self._graph

    async def prepare_state(
        self, query: str, session_id: str, chat_history: list[Message]
    ) -> GraphState:
        return GraphState(query=query, chat_history=chat_history, session_id=session_id)

    async def close(self) -> None:
        """리소스 정리 - ElasticsearchStore 연결 종료"""
        try:
            await elasticsearch_store.close()
        except Exception as e:
            logger.error("[RAGGraph] 리소스 정리 중 오류: {}", e)
        finally:
            self._initialized = False


rag_graph = RAGGraph()
