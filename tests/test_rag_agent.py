# tests/test_rag_agent.py

import pytest
from unittest.mock import MagicMock
from langchain.schema import Document

# 테스트 대상 클래스 임포트
from src.services.rag_agent import RagAgent
# 의존성 클래스 임포트 (모방 대상)
from src.storage.elastic_store import ElasticStore
from src.clients.ollama_client import OllamaClient


@pytest.fixture
def mock_store():
    """가짜 ElasticStore (VectorStore) 객체를 생성합니다."""
    store = MagicMock(spec=ElasticStore)
    store.search_knn.return_value = [
        Document(page_content="검색된 가짜 컨텍스트입니다.", metadata={"title": "가짜 문서"})
    ]
    return store


@pytest.fixture
def mock_llm_client():
    """가짜 OllamaClient 객체를 생성합니다."""
    client = MagicMock(spec=OllamaClient)
    # HyDE 호출, RAG 호출 순서대로 다른 값 반환
    client.get_response.side_effect = [
        "HyDE가 생성한 가짜 답변입니다.",
        "Ollama가 생성한 최종 답변입니다."
    ]
    return client


def test_rag_agent_query_logic(mock_store, mock_llm_client):
    """
    RagAgent.query()의 전체 로직을 테스트합니다.
    (HyDE -> Search -> RAG 흐름 검증)
    """

    # 1. Arrange: 테스트 대상 클래스에 가짜 의존성 주입
    agent = RagAgent(vector_store=mock_store, llm_client=mock_llm_client)
    test_question = "AI의 가치 정렬 문제는?"

    # 2. Act
    final_answer = agent.query(test_question)

    # 3. Assert

    # 3-1. 최종 반환값이 Ollama의 두 번째 응답과 일치하는가?
    assert final_answer == "Ollama가 생성한 최종 답변입니다."

    # 3-2. LLM이 총 2번 호출되었는가? (HyDE, RAG)
    assert mock_llm_client.get_response.call_count == 2

    # 3-3. 첫 번째 (HyDE) 호출이 '원본 질문'을 포함했는가?
    first_call_args = mock_llm_client.get_response.call_args_list[0]
    assert test_question in first_call_args[0][0]  # 프롬프트 문자열

    # 3-4. Vector Store 검색이 'HyDE의 가짜 답변'으로 호출되었는가?
    mock_store.search_knn.assert_called_once_with(
        "HyDE가 생성한 가짜 답변입니다.",
        k=3
    )

    # 3-5. 두 번째 (RAG) 호출이 '원본 질문'과 '검색된 컨텍스트'를 포함했는가?
    second_call_args = mock_llm_client.get_response.call_args_list[1]
    rag_prompt_content = second_call_args[0][0]  # 프롬프트 문자열
    assert test_question in rag_prompt_content
    assert "검색된 가짜 컨텍스트입니다." in rag_prompt_content