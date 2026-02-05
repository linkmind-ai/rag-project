"""
RAG 서비스 레이어 모듈.

이 모듈은 API 라우터와 RAG 그래프 사이의 비즈니스 로직을 담당합니다.
- 세션 기반 대화 이력 관리
- LangGraph 워크플로우 실행 조율
- 스트리밍/비스트리밍 응답 처리
"""

import asyncio
from typing import Any, AsyncGenerator, Dict

from langchain_core.runnables import RunnableConfig

from graphs.rag_graph import rag_graph
from stores.memory_store import memory_store


class RAGService:
    """
    RAG 시스템 서비스 레이어.

    API 엔드포인트와 LangGraph 워크플로우 사이에서 비즈니스 로직을 처리합니다:
    - 대화 이력 로드/저장
    - 그래프 상태 준비 및 실행
    - 응답 포맷팅

    Attributes:
        _initialized: 초기화 완료 플래그 (Double-Checked Locking용)
        _lock: 동시성 제어를 위한 비동기 락
    """

    def __init__(self) -> None:
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """서비스 초기화"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            await rag_graph.initialize()
            self._initialized = True

    async def process_query(
        self, session_id: str, query: str, use_history: bool = True
    ) -> Dict[str, Any]:
        """
        쿼리 처리 및 응답 생성 (동기식 invoke).

        전체 RAG 파이프라인을 실행하고 완성된 응답을 반환합니다.

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
        """
        await self.initialize()

        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)

        initial_state = await rag_graph.prepare_state(
            query=query, session_id=session_id, chat_history=chat_history
        )

        graph = rag_graph.get_graph()

        # LangGraph는 dict 입력을 기대함 - Pydantic 모델을 dict로 변환
        result = await graph.ainvoke(initial_state.model_dump())

        answer = result["answer"]
        evidence_indices = result.get("evidence_indices", [])
        retrieved_docs = result.get("retrieved_docs", [])

        evidence_docs = [
            retrieved_docs[idx]
            for idx in evidence_indices
            if 0 <= idx < len(retrieved_docs)
        ]

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", answer)

        return {
            "answer": answer,
            "evidence_indices": evidence_indices,
            "evidence_docs": evidence_docs,
            "all_docs": retrieved_docs,
        }

    async def process_query_stream(
        self, session_id: str, query: str, use_history: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """쿼리-답변 스트리밍 실행"""
        await self.initialize()

        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)

        initial_state = await rag_graph.prepare_state(
            query=query, session_id=session_id, chat_history=chat_history
        )

        config = RunnableConfig(
            configurable={
                "session_id": session_id,
                "thread_id": session_id,  # 추후 쓰레드ID는 수정 필요
            }
        )

        graph = rag_graph.get_graph()

        full_response = ""
        retrieved_docs = []
        evidence_indices = []

        # LangGraph는 dict 입력을 기대함 - Pydantic 모델을 dict로 변환
        async for event in graph.astream_events(
            initial_state.model_dump(), config, version="v1"
        ):
            event_type = event.get("event")
            name = event.get("name", "")

            if event_type == "on_chain_start" and name == "retrieve":
                yield {"type": "retrieve_start", "message": "문서 검색 중..."}

            elif event_type == "on_chain_end" and name == "retrieve":
                data = event.get("data", {})
                output = data.get("output", {})
                retrieved_docs = output.get("retrieved_docs", [])

                yield {
                    "type": "retrieve_end",
                    "message": f"{len(retrieved_docs)}개의 문서를 찾았습니다.",
                    "doc_count": len(retrieved_docs),
                }

            elif event_type == "on_chain_start" and name == "generate":
                yield {"type": "generate_start", "message": "답변 생성 중..."}

            elif event_type == "on_chain_stream" and name == "generate":
                data = event.get("data", {})
                chunk = data.get("chunk", {})

                if isinstance(chunk, dict):
                    answer_chunk = chunk.get("answer", "")
                elif isinstance(chunk, str):
                    answer_chunk = chunk
                else:
                    answer_chunk = str(chunk) if chunk else ""

                if answer_chunk:
                    full_response += answer_chunk
                    yield {"type": "content", "content": answer_chunk}

            elif event_type == "on_chain_end" and name == "generate":
                data = event.get("data", {})
                output = data.get("output", {})

                if not full_response and "answer" in output:
                    full_response = output["answer"]
                    yield {"type": "content", "content": full_response}

                yield {"type": "generate_end", "message": "답변 생성 완료"}

            elif event_type == "on_chain_start" and name == "identify_evidence":
                yield {"type": "evidence_start", "message": "답변 근거 분석 중..."}

            elif event_type == "on_chain_end" and name == "identify_evidence":
                data = event.get("data", {})
                output = data.get("output", {})
                evidence_indices = output.get("evidence_indices", [])

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

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", full_response)

        yield {
            "type": "done",
            "full_response": full_response,
            "evidence_indices": evidence_indices,
        }


rag_service = RAGService()
