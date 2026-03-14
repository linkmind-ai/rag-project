"""
Identify Evidence 노드 품질 검증 테스트

LLM 호출을 Mock하여 근거 문서 식별 시나리오를 시뮬레이션합니다.
- JSON 형식 응답 파싱
- 마크다운 코드블록 형식 파싱
- 잘못된 형식 fallback 처리
- 빈 문서/범위 벗어난 인덱스 처리
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from graphs.rag_graph import RAGGraph
from models.state import Document, GraphState


class TestParseEvidenceResponse:
    """_parse_evidence_response 메서드 테스트"""

    @pytest.fixture
    def rag_graph(self) -> RAGGraph:
        """RAGGraph 인스턴스 생성"""
        graph = RAGGraph()
        graph._initialized = True
        return graph

    @pytest.mark.asyncio
    async def test_parse_json_object_format(self, rag_graph: RAGGraph) -> None:
        """JSON 객체 형식 파싱 테스트

        입력: {"evidence_indices": [0, 1, 2]}
        """
        response = '{"evidence_indices": [0, 1, 2]}'
        result = await rag_graph._parse_evidence_response(response)

        assert result == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_parse_json_array_format(self, rag_graph: RAGGraph) -> None:
        """JSON 배열 형식 파싱 테스트

        입력: [0, 2, 3]
        """
        response = "[0, 2, 3]"
        result = await rag_graph._parse_evidence_response(response)

        assert result == [0, 2, 3]

    @pytest.mark.asyncio
    async def test_parse_markdown_codeblock_format(self, rag_graph: RAGGraph) -> None:
        """마크다운 코드블록 형식 파싱 테스트

        입력: ```json\n{"evidence_indices": [1, 3]}\n```
        """
        response = '```json\n{"evidence_indices": [1, 3]}\n```'
        result = await rag_graph._parse_evidence_response(response)

        assert result == [1, 3]

    @pytest.mark.asyncio
    async def test_parse_with_extra_text(self, rag_graph: RAGGraph) -> None:
        """추가 텍스트가 포함된 응답 파싱 테스트

        입력: 근거 문서는 다음과 같습니다: {"evidence_indices": [0]}
        """
        response = '근거 문서는 다음과 같습니다: {"evidence_indices": [0]}'
        result = await rag_graph._parse_evidence_response(response)

        assert result == [0]

    @pytest.mark.asyncio
    async def test_parse_invalid_json_fallback(self, rag_graph: RAGGraph) -> None:
        """잘못된 JSON 형식 → 숫자 추출 fallback

        입력: 인덱스 0, 2번 문서가 근거입니다
        """
        response = "인덱스 0, 2번 문서가 근거입니다"
        result = await rag_graph._parse_evidence_response(response)

        assert 0 in result
        assert 2 in result

    @pytest.mark.asyncio
    async def test_parse_empty_response(self, rag_graph: RAGGraph) -> None:
        """빈 응답 처리 테스트"""
        response = ""
        result = await rag_graph._parse_evidence_response(response)

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_no_numbers_response(self, rag_graph: RAGGraph) -> None:
        """숫자가 없는 응답 처리 테스트"""
        response = "근거 문서를 찾을 수 없습니다."
        result = await rag_graph._parse_evidence_response(response)

        assert result == []

    @pytest.mark.asyncio
    async def test_parse_string_indices(self, rag_graph: RAGGraph) -> None:
        """문자열 인덱스를 정수로 변환 테스트

        입력: {"evidence_indices": ["0", "1"]}
        """
        response = '{"evidence_indices": ["0", "1"]}'
        result = await rag_graph._parse_evidence_response(response)

        assert result == [0, 1]
        assert all(isinstance(idx, int) for idx in result)


class TestIdentifyEvidenceNode:
    """_identify_evidence_node 메서드 테스트"""

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
                content="단일성 정체감 장애는 인격이 하나인 상태입니다.",
                metadata={"source": "test"},
                doc_id="doc_0",
            ),
            Document(
                content="저자는 결혼하지 말라고 조언합니다.",
                metadata={"source": "test"},
                doc_id="doc_1",
            ),
            Document(
                content="관찰력은 가설을 갱신할 수 있어야 좋은 관찰입니다.",
                metadata={"source": "test"},
                doc_id="doc_2",
            ),
        ]

    def _create_state(
        self,
        query: str,
        answer: str,
        documents: List[Document],
    ) -> GraphState:
        """테스트용 GraphState 생성"""
        return GraphState(
            query=query,
            answer=answer,
            retrieved_docs=documents,
            chat_history=[],
            session_id="test_session",
        )

    @pytest.mark.asyncio
    async def test_identify_evidence_with_valid_indices(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """정상적인 근거 인덱스 반환 테스트"""
        query = "단일성 정체감 장애란?"
        answer = "인격이 하나인 상태입니다."
        state = self._create_state(query, answer, sample_documents)

        mock_response = '{"evidence_indices": [0]}'

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._identify_evidence_node(state)

        assert "evidence_indices" in result
        assert result["evidence_indices"] == [0]

    @pytest.mark.asyncio
    async def test_identify_evidence_multiple_indices(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """여러 근거 문서 인덱스 반환 테스트"""
        query = "단일성 정체감 장애와 저자의 조언은?"
        answer = "인격이 하나인 상태이며, 저자는 결혼하지 말라고 조언합니다."
        state = self._create_state(query, answer, sample_documents)

        mock_response = '{"evidence_indices": [0, 1]}'

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._identify_evidence_node(state)

        assert result["evidence_indices"] == [0, 1]

    @pytest.mark.asyncio
    async def test_identify_evidence_empty_documents(self, rag_graph: RAGGraph) -> None:
        """빈 문서 목록 처리 테스트"""
        state = self._create_state("질문", "답변", [])

        result = await rag_graph._identify_evidence_node(state)

        assert result["evidence_indices"] == []

    @pytest.mark.asyncio
    async def test_identify_evidence_filters_invalid_indices(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """범위 벗어난 인덱스 필터링 테스트

        문서가 3개인데 인덱스 [0, 5, 10]을 반환하면 [0]만 유효
        """
        query = "테스트"
        answer = "테스트 답변"
        state = self._create_state(query, answer, sample_documents)

        mock_response = '{"evidence_indices": [0, 5, 10]}'

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._identify_evidence_node(state)

        assert result["evidence_indices"] == [0]

    @pytest.mark.asyncio
    async def test_identify_evidence_filters_negative_indices(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """음수 인덱스 필터링 테스트"""
        query = "테스트"
        answer = "테스트 답변"
        state = self._create_state(query, answer, sample_documents)

        mock_response = '{"evidence_indices": [-1, 0, 1]}'

        with patch("asyncio.to_thread", return_value=mock_response):
            result = await rag_graph._identify_evidence_node(state)

        assert -1 not in result["evidence_indices"]
        assert result["evidence_indices"] == [0, 1]

    @pytest.mark.asyncio
    async def test_identify_evidence_llm_exception_fallback(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """LLM 예외 발생 시 모든 문서 인덱스 반환 (fallback)"""
        query = "테스트"
        answer = "테스트 답변"
        state = self._create_state(query, answer, sample_documents)

        with patch("asyncio.to_thread", side_effect=Exception("LLM 오류")):
            result = await rag_graph._identify_evidence_node(state)

        # fallback: 모든 문서 인덱스 반환
        assert result["evidence_indices"] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_identify_evidence_documents_text_format(
        self, rag_graph: RAGGraph, sample_documents: List[Document]
    ) -> None:
        """문서 텍스트 포맷 검증

        [인덱스 0], [인덱스 1], ... 형식으로 전달되는지 확인
        """
        query = "테스트"
        answer = "테스트 답변"
        state = self._create_state(query, answer, sample_documents)

        captured_input: Dict[str, Any] = {}

        def capture_invoke(input_data: Dict[str, Any]) -> str:
            captured_input.update(input_data)
            return '{"evidence_indices": [0]}'

        with patch(
            "asyncio.to_thread", side_effect=lambda fn, data: capture_invoke(data)
        ):
            await rag_graph._identify_evidence_node(state)

        assert "documents" in captured_input
        assert "[인덱스 0]" in captured_input["documents"]
        assert "[인덱스 1]" in captured_input["documents"]
        assert "[인덱스 2]" in captured_input["documents"]


class TestEvidencePromptQuality:
    """근거 추출 프롬프트 품질 테스트"""

    def test_evidence_prompt_has_query_placeholder(self) -> None:
        """프롬프트에 query 플레이스홀더 존재 확인"""
        from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT

        prompt_str = str(_GET_EVIDENCE_PROMPT)
        assert "query" in prompt_str.lower()

    def test_evidence_prompt_has_answer_placeholder(self) -> None:
        """프롬프트에 answer 플레이스홀더 존재 확인"""
        from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT

        prompt_str = str(_GET_EVIDENCE_PROMPT)
        assert "answer" in prompt_str.lower()

    def test_evidence_prompt_has_documents_placeholder(self) -> None:
        """프롬프트에 documents 플레이스홀더 존재 확인"""
        from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT

        prompt_str = str(_GET_EVIDENCE_PROMPT)
        assert "documents" in prompt_str.lower()

    def test_evidence_prompt_specifies_json_format(self) -> None:
        """프롬프트에 JSON 형식 지시 존재 확인"""
        from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT

        prompt_str = str(_GET_EVIDENCE_PROMPT)
        assert "json" in prompt_str.lower()
        assert "evidence_indices" in prompt_str


class TestEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture
    def rag_graph(self) -> RAGGraph:
        """RAGGraph 인스턴스 생성"""
        graph = RAGGraph()
        graph._initialized = True
        return graph

    @pytest.mark.asyncio
    async def test_parse_deeply_nested_json(self, rag_graph: RAGGraph) -> None:
        """중첩된 JSON 구조 처리"""
        response = '{"result": {"evidence_indices": [0, 1]}}'
        result = await rag_graph._parse_evidence_response(response)

        # 최상위에 evidence_indices가 없으므로 빈 배열 또는 fallback
        # 현재 구현에서는 빈 배열 반환
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_parse_unicode_response(self, rag_graph: RAGGraph) -> None:
        """유니코드 포함 응답 처리"""
        response = '{"evidence_indices": [0, 1]} // 근거 문서 인덱스입니다'
        result = await rag_graph._parse_evidence_response(response)

        assert 0 in result
        assert 1 in result

    @pytest.mark.asyncio
    async def test_parse_float_indices(self, rag_graph: RAGGraph) -> None:
        """실수 인덱스를 정수로 변환"""
        response = '{"evidence_indices": [0.0, 1.5, 2.9]}'
        result = await rag_graph._parse_evidence_response(response)

        assert all(isinstance(idx, int) for idx in result)

    @pytest.mark.asyncio
    async def test_parse_duplicate_indices(self, rag_graph: RAGGraph) -> None:
        """중복 인덱스 처리 (현재 구현은 중복 허용)"""
        response = '{"evidence_indices": [0, 0, 1, 1]}'
        result = await rag_graph._parse_evidence_response(response)

        assert result == [0, 0, 1, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
