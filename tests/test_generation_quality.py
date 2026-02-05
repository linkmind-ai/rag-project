"""
Generate 노드 품질 검증 테스트

LLM 호출을 Mock하여 다양한 응답 시나리오를 시뮬레이션합니다.
- 컨텍스트에 정답이 있는 경우
- 컨텍스트가 질문과 관련 없는 경우
- 컨텍스트가 비어 있는 경우
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from graphs.rag_graph import RAGGraph
from models.state import Document, GraphState, Message


class TestGenerateNode:
    """Generate 노드 테스트 클래스"""

    @pytest.fixture
    def rag_graph(self) -> RAGGraph:
        """RAGGraph 인스턴스 생성"""
        graph = RAGGraph()
        graph._initialized = True
        graph._llm = MagicMock()
        return graph

    @pytest.fixture
    def sample_documents(self) -> List[Document]:
        """테스트용 샘플 문서"""
        return [
            Document(
                content="단일성 정체감의 장애는 인격이 하나라 합리성이라는 단일 필터로 세상을 본다는 점입니다.",
                metadata={"source": "test", "page_id": "test_page_1"},
                doc_id="doc_1",
            ),
            Document(
                content="저자는 이런 사람을 존중하지만 절대 결혼하지 말라고 권합니다.",
                metadata={"source": "test", "page_id": "test_page_1"},
                doc_id="doc_2",
            ),
        ]

    @pytest.fixture
    def unrelated_documents(self) -> List[Document]:
        """질문과 관련 없는 문서"""
        return [
            Document(
                content="오늘 날씨가 매우 좋습니다. 산책하기 좋은 날입니다.",
                metadata={"source": "weather", "page_id": "weather_1"},
                doc_id="doc_weather",
            ),
        ]

    @pytest.fixture
    def empty_documents(self) -> List[Document]:
        """빈 문서 리스트"""
        return []

    def _create_state(
        self,
        query: str,
        documents: List[Document],
        chat_history: List[Message] = None,
    ) -> GraphState:
        """테스트용 GraphState 생성"""
        return GraphState(
            query=query,
            retrieved_docs=documents,
            chat_history=chat_history or [],
            session_id="test_session",
        )

    @pytest.mark.asyncio
    async def test_generate_with_relevant_context(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """컨텍스트에 정답이 있는 경우 테스트

        검증 항목:
        - 정확한 정보 포함 여부
        - 인용/출처 언급 여부
        """
        query = "단일성 정체감 장애의 특징은 무엇인가요?"
        state = self._create_state(query, sample_documents)

        mock_response = (
            "단일성 정체감의 장애는 인격이 하나라서 세상 모든 일을 "
            "합리성이라는 단일 필터로 본다는 특징이 있습니다. "
            "[Document 1 참조]"
        )

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._generate_node(state)

        assert "answer" in result
        answer = result["answer"]

        # 정확한 정보 포함 여부
        assert "합리성" in answer or "단일 필터" in answer or "인격" in answer
        # 인용 포함 여부
        assert "Document" in answer or "참조" in answer

    @pytest.mark.asyncio
    async def test_generate_with_unrelated_context(
        self, rag_graph: RAGGraph, unrelated_documents: List[Document]
    ) -> None:
        """컨텍스트가 질문과 관련 없는 경우 테스트

        검증 항목:
        - "정보가 부족하다" 또는 "답할 수 없다" 응답 여부
        """
        query = "단일성 정체감 장애의 특징은 무엇인가요?"
        state = self._create_state(query, unrelated_documents)

        mock_response = (
            "제공된 컨텍스트에는 단일성 정체감 장애에 대한 정보가 없습니다. "
            "해당 질문에 답변하기 위한 충분한 정보가 부족합니다."
        )

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._generate_node(state)

        assert "answer" in result
        answer = result["answer"]

        # 정보 부족 관련 문구 포함 여부
        insufficient_keywords = ["정보가 없", "정보가 부족", "답할 수 없", "모르"]
        has_insufficient_response = any(kw in answer for kw in insufficient_keywords)
        assert has_insufficient_response, f"Expected insufficient info response, got: {answer}"

    @pytest.mark.asyncio
    async def test_generate_with_empty_context(
        self, rag_graph: RAGGraph, empty_documents: List[Document]
    ) -> None:
        """컨텍스트가 비어 있는 경우 테스트

        검증 항목:
        - 예외 처리 또는 가이드 문구 출력
        """
        query = "단일성 정체감 장애의 특징은 무엇인가요?"
        state = self._create_state(query, empty_documents)

        mock_response = (
            "검색된 문서가 없어 질문에 답변드리기 어렵습니다. "
            "다른 키워드로 질문해 주시거나 관련 문서를 추가해 주세요."
        )

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._generate_node(state)

        assert "answer" in result
        answer = result["answer"]

        # 가이드 문구 포함 여부
        guide_keywords = ["문서가 없", "답변드리기 어렵", "다른 키워드", "추가해"]
        has_guide_response = any(kw in answer for kw in guide_keywords)
        assert has_guide_response, f"Expected guide response, got: {answer}"

    @pytest.mark.asyncio
    async def test_generate_with_chat_history(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """대화 이력이 있는 경우 테스트

        검증 항목:
        - chat_with_history_prompt 사용 여부
        - 이전 대화 컨텍스트 반영 여부
        """
        query = "저자의 조언은 무엇인가요?"
        chat_history = [
            Message(role="user", content="단일성 정체감 장애가 뭔가요?"),
            Message(role="assistant", content="인격이 하나인 상태를 말합니다."),
        ]
        state = self._create_state(query, sample_documents, chat_history)

        mock_response = (
            "이전 대화에서 언급한 단일성 정체감 장애에 대해, "
            "저자는 그런 사람을 존중하되 절대 결혼하지 말라고 권합니다."
        )

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._generate_node(state)

        assert "answer" in result
        answer = result["answer"]
        assert "결혼" in answer or "저자" in answer

    @pytest.mark.asyncio
    async def test_build_context_format(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """_build_context 메서드의 포맷 검증

        검증 항목:
        - Document 번호 포함
        - 문서 내용 포함
        """
        context = rag_graph._build_context(sample_documents)

        assert "Document 1:" in context
        assert "Document 2:" in context
        assert "단일성 정체감" in context
        assert "저자는" in context

    @pytest.mark.asyncio
    async def test_generate_handles_llm_exception(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """LLM 호출 실패 시 예외 처리 테스트"""
        query = "테스트 질문"
        state = self._create_state(query, sample_documents)

        async def raise_exception(*args: Any, **kwargs: Any) -> None:
            raise Exception("LLM 호출 실패")

        with patch("asyncio.to_thread", side_effect=Exception("LLM 호출 실패")):
            with pytest.raises(Exception) as exc_info:
                await rag_graph._generate_node(state)

            assert "LLM 호출 실패" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_context_string(
        self, rag_graph: RAGGraph, empty_documents: List[Document]
    ) -> None:
        """빈 문서 리스트에서 컨텍스트 문자열 검증"""
        context = rag_graph._build_context(empty_documents)
        assert context == ""


class TestPromptQuality:
    """프롬프트 품질 테스트"""

    def test_chat_prompt_has_context_placeholder(self) -> None:
        """프롬프트에 context 플레이스홀더 존재 확인"""
        from prompts.chat_prompt import _CHAT_PROMPT

        prompt_str = str(_CHAT_PROMPT)
        assert "context" in prompt_str.lower()

    def test_chat_prompt_has_query_placeholder(self) -> None:
        """프롬프트에 query 플레이스홀더 존재 확인"""
        from prompts.chat_prompt import _CHAT_PROMPT

        prompt_str = str(_CHAT_PROMPT)
        assert "query" in prompt_str.lower()

    def test_chat_prompt_instructs_uncertainty(self) -> None:
        """프롬프트에 불확실성 처리 지침 존재 확인"""
        from prompts.chat_prompt import _CHAT_PROMPT

        prompt_str = str(_CHAT_PROMPT)
        assert "모른다" in prompt_str or "확실하지 않" in prompt_str


class TestGenerateNodeIntegration:
    """Generate 노드 통합 테스트 (실제 프롬프트 사용)"""

    @pytest.fixture
    def rag_graph(self) -> RAGGraph:
        """RAGGraph 인스턴스 생성"""
        return RAGGraph()

    @pytest.mark.asyncio
    async def test_prompt_receives_correct_context(
        self, rag_graph: RAGGraph
    ) -> None:
        """프롬프트에 올바른 컨텍스트가 전달되는지 확인"""
        documents = [
            Document(
                content="테스트 문서 내용입니다.",
                metadata={"source": "test"},
                doc_id="test_doc",
            ),
        ]
        state = GraphState(
            query="테스트 질문",
            retrieved_docs=documents,
            chat_history=[],
            session_id="test",
        )

        captured_input: Dict[str, Any] = {}

        def capture_invoke(input_data: Dict[str, Any]) -> str:
            captured_input.update(input_data)
            return "테스트 응답"

        rag_graph._initialized = True
        rag_graph._llm = MagicMock()

        with patch("asyncio.to_thread", side_effect=lambda fn, data: capture_invoke(data)):
            await rag_graph._generate_node(state)

        assert "context" in captured_input
        assert "테스트 문서 내용" in captured_input["context"]
        assert "query" in captured_input
        assert captured_input["query"] == "테스트 질문"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])