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

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from common.config import settings
from models.state import Document, GraphState, Message
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT
from stores.vector_store import elasticsearch_store

# ========== 한국어 불용어 (Stopwords) ==========
# 키워드 기반 근거 식별 시 노이즈를 제거하기 위한 불용어 집합.
# 조사, 접속사, 대명사 등 의미적 가치가 낮은 단어들을 필터링하여
# 핵심 키워드 매칭의 정확도를 높입니다.
# ==============================================
_KOREAN_STOPWORDS: Set[str] = {
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
        self._llm: Optional[Ollama] = None
        self._graph = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT
        self.get_evidence_prompt = _GET_EVIDENCE_PROMPT

    async def __aenter__(self) -> "RAGGraph":
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """비동기 컨텍스트 매니저 종료 - 리소스 정리"""
        await self.close()

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=1.0,
            )

            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        """그래프 구조 빌드"""
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("identify_evidence", self._identify_evidence_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "identify_evidence")
        workflow.add_edge("identify_evidence", END)

        self._graph = workflow.compile()

    async def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """검색 노드"""
        query = state.query

        context = await elasticsearch_store.hybrid_search(
            query=query, k=settings.TOP_K_RESULTS, vector_weight=0.5
        )

        return {"retrieved_docs": context.documents}

    def _prepare_messages(self, chat_history: List[Message]) -> List[Any]:
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

    def _build_context(self, retrieved_docs: List[Document]) -> str:
        """검색 문서 텍스트 변환"""
        context_text = "\n\n".join(
            [f"Document {i+1}:\n{doc.content}" for i, doc in enumerate(retrieved_docs)]
        )

        return context_text

    async def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        """응답 생성 노드"""
        query = state.query
        retrieved_docs = state.retrieved_docs
        chat_history = state.chat_history

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

    def _extract_keywords(self, text: str, min_length: int = 2) -> Set[str]:
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
    ) -> Tuple[float, Set[str]]:
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
        retrieved_docs: List[Document],
        threshold: float = 0.1,
    ) -> List[Tuple[int, float, Set[str]]]:
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

    async def _parse_evidence_response(self, response: str) -> List[int]:
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
            print(f"근거 응답 파싱 오류: {e}")
            print(f"원본 응답: {response}")

            numbers = re.findall(r"\d+", response)
            if numbers:
                return [int(n) for n in numbers]

            return []

    async def _identify_evidence_node(self, state: GraphState) -> Dict[str, Any]:
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

        # ========== 1단계: 키워드 기반 후보 추출 ==========
        keyword_candidates = self._get_keyword_based_evidence(
            answer, retrieved_docs, threshold=0.1
        )
        keyword_indices = {idx for idx, _, _ in keyword_candidates}

        print(f"[N3] 키워드 기반 후보: {keyword_candidates[:5]}")

        # ========== 2단계: LLM 기반 근거 식별 ==========
        documents_text = "\n\n".join(
            [
                f"[인덱스 {i}]\n제목: {doc.metadata.get('title', '제목 없음')}\n내용: {doc.content[:500]}"
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
            print(f"[N3] LLM 선택 인덱스: {llm_indices}")

            # ========== 3단계: 하이브리드 결합 ==========
            final_indices_set = set(llm_indices)

            # 3-1. 키워드 일치도가 높은 문서 보강 (LLM이 놓친 경우 복구)
            # 일치도 20% 이상이면 답변에 실제로 사용된 것으로 간주
            for idx, overlap_ratio, matched in keyword_candidates:
                if overlap_ratio >= 0.2:
                    if idx not in final_indices_set:
                        print(
                            f"[N3] 키워드 기반 보강: 문서 {idx} "
                            f"(일치도: {overlap_ratio:.2%}, 키워드: {list(matched)[:5]})"
                        )
                    final_indices_set.add(idx)

            # 3-2. LLM이 선택했으나 키워드 일치도가 매우 낮은 문서 제거
            # LLM 환각으로 잘못 선택된 문서 필터링 (일치도 5% 미만)
            for idx in list(final_indices_set):
                if idx not in keyword_indices:
                    overlap, _ = self._calculate_keyword_overlap(
                        answer, retrieved_docs[idx].content
                    )
                    if overlap < 0.05:
                        print(
                            f"[N3] 낮은 일치도로 제거: 문서 {idx} (일치도: {overlap:.2%})"
                        )
                        final_indices_set.discard(idx)

            final_indices = sorted(final_indices_set)
            print(f"[N3] 최종 evidence_indices: {final_indices}")

            return {"evidence_indices": final_indices}

        except Exception as e:
            print(f"근거 문서 식별 오류: {e}")
            # Fallback: LLM 실패 시 키워드 기반 결과만 사용
            fallback_indices = [
                idx for idx, ratio, _ in keyword_candidates if ratio >= 0.15
            ]
            return {"evidence_indices": fallback_indices if fallback_indices else [0]}

    def get_graph(self):
        """컴파일된 그래프 반환"""
        return self._graph

    async def prepare_state(
        self, query: str, session_id: str, chat_history: List[Message]
    ) -> GraphState:
        return GraphState(query=query, chat_history=chat_history, session_id=session_id)

    async def close(self) -> None:
        """리소스 정리 - ElasticsearchStore 연결 종료"""
        try:
            await elasticsearch_store.close()
        except Exception as e:
            print(f"[RAGGraph] 리소스 정리 중 오류: {e}")
        finally:
            self._initialized = False


rag_graph = RAGGraph()
