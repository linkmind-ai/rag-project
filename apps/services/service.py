"""
RAG 서비스 레이어 모듈.

이 모듈은 API 라우터와 RAG 그래프 사이의 비즈니스 로직을 담당합니다.
- 세션 기반 대화 이력 관리
- LangGraph 워크플로우 실행 조율
- 스트리밍/비스트리밍 응답 처리

┌─────────────────────────────────────────────────────────────────────────┐
│                    RAGService 스트리밍 이벤트 흐름                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [Client Request]                                                      │
│         │                                                               │
│         ▼                                                               │
│   ┌─────────────┐                                                       │
│   │ initialize  │ ◀── Double-Checked Locking 패턴                       │
│   └──────┬──────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐     ┌─────────────────────────────────────────┐      │
│   │ RAGGraph    │────▶│  이벤트 타입 (process_query_stream)      │      │
│   │ astream_    │     ├─────────────────────────────────────────┤      │
│   │ events()    │     │ • retrieve_start  : 검색 시작 알림       │      │
│   └─────────────┘     │ • retrieve_end    : 검색 완료 + 문서 수  │      │
│                       │ • generate_start  : 생성 시작 알림       │      │
│                       │ • content         : 답변 토큰 스트리밍   │      │
│                       │ • generate_end    : 생성 완료 알림       │      │
│                       │ • evidence_start  : 근거 분석 시작       │      │
│                       │ • evidence_end    : 근거 인덱스 + 문서   │      │
│                       │ • error           : 에러 발생 시 (신규)  │      │
│                       │ • done            : 전체 완료 + 요약     │      │
│                       └─────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Note:
    - 모든 이벤트는 Dict[str, Any] 형태로 yield됨
    - 'type' 키로 이벤트 종류 구분, 클라이언트는 이를 기준으로 UI 업데이트
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from common.logger import logger
from graphs.rag_graph import rag_graph
from graphs.corrective_rag_graph import corrective_rag_graph
from langchain_core.runnables import RunnableConfig
from stores.memory_store import memory_store


class RAGService:
    """
    RAG 시스템 서비스 레이어.

    API 엔드포인트와 LangGraph 워크플로우 사이에서 비즈니스 로직을 처리합니다:
    - 대화 이력 로드/저장
    - 그래프 상태 준비 및 실행
    - 응답 포맷팅

    ┌─────────────────────────────────────────────────────────────────┐
    │                    RAGService 책임 범위                          │
    ├─────────────────────────────────────────────────────────────────┤
    │ 1. 초기화 관리                                                   │
    │    └─ Double-Checked Locking으로 RAGGraph 단일 초기화 보장       │
    │                                                                 │
    │ 2. 대화 이력 조율                                                │
    │    └─ memory_store에서 이전 대화 로드 → 그래프에 전달            │
    │                                                                 │
    │ 3. 그래프 실행 모드 선택                                         │
    │    ├─ process_query: ainvoke (동기식, 전체 응답 반환)           │
    │    └─ process_query_stream: astream_events (비동기 스트리밍)    │
    │                                                                 │
    │ 4. 결과 후처리                                                   │
    │    └─ 대화 이력 저장, 근거 문서 추출, 응답 포맷팅                │
    └─────────────────────────────────────────────────────────────────┘

    Attributes:
        _initialized: 초기화 완료 플래그 (Double-Checked Locking용)
        _lock: 동시성 제어를 위한 비동기 락
    """

    def __init__(self) -> None:
        """RAGService 인스턴스 초기화."""
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        서비스 초기화 (Double-Checked Locking 패턴).

        ┌─────────────────────────────────────────────────────────────────┐
        │              Double-Checked Locking 상세 흐름                    │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │   [요청 1] ──┐                   [요청 2] ──┐                    │
        │              │                              │                   │
        │              ▼                              ▼                   │
        │        ┌──────────┐                  ┌──────────┐               │
        │        │ 1차 체크  │                  │ 1차 체크  │               │
        │        │ 초기화됨? │                  │ 초기화됨? │               │
        │        └────┬─────┘                  └────┬─────┘               │
        │             │ No                          │ No                  │
        │             ▼                             ▼                     │
        │        ┌──────────┐                  ┌──────────┐               │
        │        │ 락 획득   │ ◀─── 대기 ─────│ 락 대기   │               │
        │        └────┬─────┘                  └────┬─────┘               │
        │             │                             │                     │
        │             ▼                             │                     │
        │        ┌──────────┐                       │                     │
        │        │ 2차 체크  │                       │                     │
        │        │ 초기화됨? │                       │                     │
        │        └────┬─────┘                       │                     │
        │             │ No                          │                     │
        │             ▼                             │                     │
        │        ┌──────────┐                       │                     │
        │        │ 실제 초기화│                       │                     │
        │        │ rag_graph │                       │                     │
        │        │ .init()   │                       │                     │
        │        └────┬─────┘                       │                     │
        │             │                             │                     │
        │             ▼                             ▼                     │
        │        ┌──────────┐                  ┌──────────┐               │
        │        │ 락 해제   │ ────────────▶   │ 락 획득   │               │
        │        └──────────┘                  └────┬─────┘               │
        │                                          │                     │
        │                                          ▼                     │
        │                                     ┌──────────┐               │
        │                                     │ 2차 체크  │               │
        │                                     │ 초기화됨? │               │
        │                                     └────┬─────┘               │
        │                                          │ Yes (이미 완료!)     │
        │                                          ▼                     │
        │                                     ┌──────────┐               │
        │                                     │ 즉시 반환 │               │
        │                                     └──────────┘               │
        │                                                                 │
        │   핵심: 2차 체크가 없으면 요청 2도 중복 초기화를 시도하게 됨     │
        └─────────────────────────────────────────────────────────────────┘

        Note:
            - asyncio.Lock()은 코루틴 간 동기화용 (threading.Lock과 다름)
            - 1차 체크: 락 없이 빠른 경로 제공 (대부분의 호출은 여기서 반환)
            - 2차 체크: 락 대기 중 다른 코루틴이 초기화했을 수 있으므로 재확인
        """
        # 1차 체크 (락 없이) - 이미 초기화된 경우 빠른 반환
        if self._initialized:
            return

        # 락 획득 - 동시 초기화 시도 방지
        async with self._lock:
            # 2차 체크 (락 안에서) - 대기 중 다른 코루틴이 초기화했을 수 있음
            if self._initialized:
                return

            # 실제 초기화 수행
            logger.debug("[RAGService] 초기화 시작...")
            await corrective_rag_graph.initialize()
            self._initialized = True
            logger.debug("[RAGService] 초기화 완료")

    async def process_query(
        self, session_id: str, query: str, use_history: bool = True
    ) -> dict[str, Any]:
        """
        쿼리 처리 및 응답 생성 (동기식 invoke).

        전체 RAG 파이프라인을 실행하고 완성된 응답을 반환합니다.
        스트리밍이 필요 없는 경우(예: 배치 처리, 테스트)에 사용합니다.

        ┌─────────────────────────────────────────────────────────────────┐
        │                    process_query 실행 흐름                       │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │   1. 초기화 확인 (Double-Checked Locking)                        │
        │                    │                                            │
        │                    ▼                                            │
        │   2. 대화 이력 로드 (use_history=True인 경우)                    │
        │      └─ memory_store.get_recent_messages(session_id)            │
        │                    │                                            │
        │                    ▼                                            │
        │   3. 그래프 상태 준비                                            │
        │      └─ rag_graph.prepare_state(query, session_id, history)     │
        │                    │                                            │
        │                    ▼                                            │
        │   4. Pydantic → Dict 변환 (model_dump)                          │
        │      └─ LangGraph는 dict 입력만 받음 (Pydantic 직접 전달 불가)   │
        │                    │                                            │
        │                    ▼                                            │
        │   5. 그래프 실행 (ainvoke)                                       │
        │      └─ retrieve → generate → identify_evidence                 │
        │                    │                                            │
        │                    ▼                                            │
        │   6. 결과 추출 및 근거 문서 필터링                               │
        │      └─ evidence_indices 범위 검증 후 문서 추출                  │
        │                    │                                            │
        │                    ▼                                            │
        │   7. 대화 이력 저장                                              │
        │      └─ user 메시지 + assistant 응답 모두 저장                   │
        │                    │                                            │
        │                    ▼                                            │
        │   8. 결과 반환                                                   │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘

        Args:
            session_id: 세션 식별자 (대화 이력 관리용)
            query: 사용자 질의 문자열
            use_history: 대화 이력 사용 여부 (기본값 True)

        Returns:
            Dict containing:
                - answer: LLM 생성 응답
                - evidence_indices: 근거 문서 인덱스 리스트
                - evidence_docs: 근거 문서 객체 리스트
                - all_docs: 검색된 전체 문서 리스트
                - elapsed_time: 처리 소요 시간 (초)

        Raises:
            Exception: 그래프 실행 중 발생한 모든 예외
        """
        # 처리 시간 측정 시작
        start_time = time.time()
        await self.initialize()

        # 대화 이력 로드 (선택적)
        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)
            logger.debug(
                f"[RAGService] session={session_id}, 이력 {len(chat_history)}개 로드"
            )

        # 그래프 초기 상태 준비
        initial_state = await corrective_rag_graph.prepare_state(
            query=query, session_id=session_id, chat_history=chat_history
        )

        graph = corrective_rag_graph.get_graph()

        # ┌─────────────────────────────────────────────────────────────┐
        # │            Pydantic → Dict 변환 (model_dump) 필요성          │
        # ├─────────────────────────────────────────────────────────────┤
        # │                                                             │
        # │ LangGraph의 StateGraph는 내부적으로 dict 기반으로 동작:      │
        # │                                                             │
        # │   graph.ainvoke(state)  ← state는 반드시 dict이어야 함      │
        # │                                                             │
        # │ Pydantic 모델을 직접 전달하면:                               │
        # │   - TypeError: 'GraphState' object is not subscriptable    │
        # │   - state["key"] 접근 시 실패                               │
        # │                                                             │
        # │ 해결책: model_dump()로 dict 변환 후 전달                     │
        # │   - Pydantic v2: model_dump() (v1의 .dict() 대체)          │
        # │   - 타입 안전성 유지하면서 LangGraph 호환성 확보             │
        # │                                                             │
        # └─────────────────────────────────────────────────────────────┘
        result = await graph.ainvoke(initial_state.model_dump())

        # 결과 추출
        answer = result["answer"]
        evidence_indices = result.get("evidence_indices", [])
        retrieved_docs = result.get("retrieved_docs", [])

        # 근거 문서 필터링 (인덱스 범위 검증)
        evidence_docs = [
            retrieved_docs[idx]
            for idx in evidence_indices
            if 0 <= idx < len(retrieved_docs)
        ]

        # 대화 이력 저장 (user + assistant 모두)
        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", answer)

        # 처리 시간 계산
        elapsed_time = time.time() - start_time
        logger.info(
            f"[RAGService] process_query 완료: session={session_id}, "
            f"docs={len(retrieved_docs)}, evidence={len(evidence_indices)}, "
            f"elapsed={elapsed_time:.2f}s"
        )

        return {
            "answer": answer,
            "evidence_indices": evidence_indices,
            "evidence_docs": evidence_docs,
            "all_docs": retrieved_docs,
            "elapsed_time": elapsed_time,
        }

    async def process_query_stream(
        self, session_id: str, query: str, use_history: bool = True
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        쿼리-답변 스트리밍 실행.

        LangGraph의 astream_events를 활용하여 각 노드의 실행 상태를
        실시간으로 클라이언트에 전달합니다. SSE(Server-Sent Events) 또는
        WebSocket 엔드포인트에서 사용합니다.

        ┌─────────────────────────────────────────────────────────────────┐
        │                스트리밍 이벤트 매핑 상세                          │
        ├─────────────────────────────────────────────────────────────────┤
        │                                                                 │
        │   LangGraph 이벤트              →    클라이언트 이벤트           │
        │   ─────────────────────────────────────────────────────────    │
        │                                                                 │
        │   on_chain_start + "retrieve"   →    retrieve_start            │
        │   │                                  {"type": "retrieve_start", │
        │   │                                   "message": "검색 중..."}  │
        │   │                                                             │
        │   on_chain_end + "retrieve"     →    retrieve_end              │
        │   │                                  {"type": "retrieve_end",   │
        │   │                                   "doc_count": N}           │
        │   │                                                             │
        │   on_chain_start + "generate"   →    generate_start            │
        │   │                                  {"type": "generate_start"} │
        │   │                                                             │
        │   on_chain_stream + "generate"  →    content (토큰별)           │
        │   │                                  {"type": "content",        │
        │   │                                   "content": "답변..."}     │
        │   │   ※ 이 이벤트가 여러 번 발생 (토큰 스트리밍)                 │
        │   │                                                             │
        │   on_chain_end + "generate"     →    generate_end              │
        │   │                                  {"type": "generate_end"}   │
        │   │                                                             │
        │   on_chain_start + "identify_   →    evidence_start            │
        │   │   evidence"                      {"type": "evidence_start"} │
        │   │                                                             │
        │   on_chain_end + "identify_     →    evidence_end              │
        │       evidence"                      {"type": "evidence_end",   │
        │                                       "evidence_indices": [...],│
        │                                       "evidence_docs": [...]}   │
        │                                                                 │
        │   [모든 노드 완료]               →    done                       │
        │                                      {"type": "done",           │
        │                                       "full_response": "...",   │
        │                                       "elapsed_time": N.NN}     │
        │                                                                 │
        │   [예외 발생 시]                 →    error (신규 추가)          │
        │                                      {"type": "error",          │
        │                                       "message": "에러 내용",   │
        │                                       "elapsed_time": N.NN}     │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘

        Args:
            session_id: 세션 식별자 (대화 이력 관리용)
            query: 사용자 질의 문자열
            use_history: 대화 이력 사용 여부 (기본값 True)

        Yields:
            Dict[str, Any]: 이벤트 타입별 데이터
                - type: 이벤트 종류 (retrieve_start, content, error, done 등)
                - 기타 필드: 이벤트별 추가 데이터

        Note:
            - 에러 발생 시 error 이벤트를 yield하고 즉시 종료
            - 정상 완료 시 done 이벤트에 전체 응답과 처리 시간 포함
            - 대화 이력은 정상 완료 시에만 저장됨
        """
        # 처리 시간 측정 시작
        start_time = time.time()

        try:
            await self.initialize()

            # 대화 이력 로드 (선택적)
            chat_history = []
            if use_history:
                chat_history = await memory_store.get_recent_messages(session_id)
                logger.debug(
                    f"[RAGService] stream session={session_id}, "
                    f"이력 {len(chat_history)}개 로드"
                )

            # 그래프 초기 상태 준비
            initial_state = await corrective_rag_graph.prepare_state(
                query=query, session_id=session_id, chat_history=chat_history
            )

            # RunnableConfig: LangGraph 실행 컨텍스트 설정
            # - session_id: 세션 추적용
            # - thread_id: LangGraph 체크포인팅용 (추후 확장 가능)
            config = RunnableConfig(
                configurable={
                    "session_id": session_id,
                    "thread_id": session_id,  # 추후 쓰레드ID는 수정 필요
                }
            )

            graph = corrective_rag_graph.get_graph()

            # 상태 변수 초기화
            full_response = ""
            retrieved_docs = []
            evidence_indices = []

            # ┌─────────────────────────────────────────────────────────┐
            # │          astream_events 사용 시 주의사항                  │
            # ├─────────────────────────────────────────────────────────┤
            # │                                                         │
            # │ 1. version="v1" 필수 지정                                │
            # │    - LangGraph 이벤트 스키마 버전                        │
            # │    - v2는 이벤트 구조가 다름                             │
            # │                                                         │
            # │ 2. model_dump() 변환 필수                                │
            # │    - Pydantic 모델 직접 전달 불가                        │
            # │    - ainvoke와 동일한 이유                               │
            # │                                                         │
            # │ 3. 이벤트 필터링 필요                                    │
            # │    - 내부 이벤트가 다수 발생                             │
            # │    - name 기준으로 관심 노드만 처리                      │
            # │                                                         │
            # └─────────────────────────────────────────────────────────┘
            async for event in graph.astream_events(
                initial_state.model_dump(), config, version="v1"
            ):
                event_type = event.get("event")
                name = event.get("name", "")

                # ─────────────────────────────────────────────────────
                # 노드별 이벤트 처리
                # ─────────────────────────────────────────────────────

                # [retrieve 노드] 문서 검색 시작
                if event_type == "on_chain_start" and name == "retrieve":
                    yield {"type": "retrieve_start", "message": "문서 검색 중..."}

                # [retrieve 노드] 문서 검색 완료
                elif event_type == "on_chain_end" and name == "retrieve":
                    data = event.get("data", {})
                    output = data.get("output", {})
                    retrieved_docs = output.get("retrieved_docs", [])

                    yield {
                        "type": "retrieve_end",
                        "message": f"{len(retrieved_docs)}개의 문서를 찾았습니다.",
                        "doc_count": len(retrieved_docs),
                    }

                # [generate 노드] 답변 생성 시작
                elif event_type == "on_chain_start" and name == "generate":
                    yield {"type": "generate_start", "message": "답변 생성 중..."}

                # [generate 노드] 답변 토큰 스트리밍
                elif event_type == "on_chain_stream" and name == "generate":
                    data = event.get("data", {})
                    chunk = data.get("chunk", {})

                    # chunk 타입에 따른 답변 추출
                    # - dict: {"answer": "..."} 형태
                    # - str: 토큰 문자열 직접
                    # - 기타: str() 변환
                    if isinstance(chunk, dict):
                        answer_chunk = chunk.get("answer", "")
                    elif isinstance(chunk, str):
                        answer_chunk = chunk
                    else:
                        answer_chunk = str(chunk) if chunk else ""

                    if answer_chunk:
                        full_response += answer_chunk
                        yield {"type": "content", "content": answer_chunk}

                # [generate 노드] 답변 생성 완료
                elif event_type == "on_chain_end" and name == "generate":
                    data = event.get("data", {})
                    output = data.get("output", {})

                    # 스트리밍 중 토큰이 누락된 경우 전체 응답으로 대체
                    if not full_response and "answer" in output:
                        full_response = output["answer"]
                        yield {"type": "content", "content": full_response}

                    yield {"type": "generate_end", "message": "답변 생성 완료"}

                # [identify_evidence 노드] 근거 분석 시작
                elif event_type == "on_chain_start" and name == "identify_evidence":
                    yield {"type": "evidence_start", "message": "답변 근거 분석 중..."}

                # [identify_evidence 노드] 근거 분석 완료
                elif event_type == "on_chain_end" and name == "identify_evidence":
                    data = event.get("data", {})
                    output = data.get("output", {})
                    evidence_indices = output.get("evidence_indices", [])

                    # 근거 문서 추출 (인덱스 범위 검증 포함)
                    evidence_docs = [
                        {
                            "index": idx,
                            "content": retrieved_docs[idx].content,
                            "metadata": retrieved_docs[idx].metadata,
                        }
                        for idx in evidence_indices
                        if 0 <= idx < len(retrieved_docs)
                    ]

                    yield {
                        "type": "evidence_end",
                        "message": f"{len(evidence_indices)}개 근거 문서 확인",
                        "evidence_indices": evidence_indices,
                        "evidence_docs": evidence_docs,
                    }

            # 정상 완료: 대화 이력 저장
            await memory_store.add_message(session_id, "user", query)
            await memory_store.add_message(session_id, "assistant", full_response)

            # 처리 시간 계산
            elapsed_time = time.time() - start_time
            logger.info(
                f"[RAGService] stream 완료: session={session_id}, "
                f"elapsed={elapsed_time:.2f}s"
            )

            # 완료 이벤트 반환
            yield {
                "type": "done",
                "full_response": full_response,
                "evidence_indices": evidence_indices,
                "elapsed_time": elapsed_time,
            }

        except Exception as e:
            # ─────────────────────────────────────────────────────────
            # 에러 처리: 에러 이벤트 yield 후 종료
            # ─────────────────────────────────────────────────────────
            elapsed_time = time.time() - start_time
            error_message = str(e)

            logger.error(
                f"[RAGService] stream 에러: session={session_id}, "
                f"error={error_message}, elapsed={elapsed_time:.2f}s"
            )

            # 클라이언트에 에러 이벤트 전달
            yield {
                "type": "error",
                "message": f"처리 중 오류가 발생했습니다: {error_message}",
                "error_detail": error_message,
                "elapsed_time": elapsed_time,
            }
            # 에러 발생 시 대화 이력은 저장하지 않음 (불완전한 응답 방지)
            return


# 싱글톤 인스턴스 (모듈 레벨)
rag_service = RAGService()
