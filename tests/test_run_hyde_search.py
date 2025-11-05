# tests/test_run_hyde_search.py

import pytest
from unittest.mock import MagicMock, ANY, call
from langchain.schema import Document

# 테스트 대상 모듈 임포트
import run_hyde_search


# ----------------- Fixtures (가짜 객체 준비) -----------------

@pytest.fixture
def mock_es_client():
    """run_hyde_pipeline에 전달될 가짜 ES Client 객체"""
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    """run_hyde_pipeline에 전달될 가짜 Tokenizer 객체"""
    return MagicMock()


@pytest.fixture
def mock_model():
    """run_hyde_pipeline에 전달될 가짜 Embedding Model 객체"""
    return MagicMock()


# ----------------- 테스트 함수 -----------------

def test_run_hyde_pipeline_logic(
        mocker, mock_es_client, mock_tokenizer, mock_model
):
    """
    run_hyde_pipeline의 전체 로직을 테스트합니다.
    - 1. HyDE (Ollama) 호출
    - 2. Embed (ES Manager) 호출
    - 3. Search (Search Pipeline) 호출
    - 4. RAG (Ollama) 호출
    이 4단계의 흐름이 올바른지 검증합니다.
    """

    # --- 1. Arrange (모든 외부 호출에 대한 가짜 응답 정의) ---

    test_question = "AI의 가치 정렬 문제는?"
    fake_hyde_answer = "HyDE가 생성한 가짜 답변입니다."
    fake_vector = [0.1, 0.2, 0.3, 0.4]  # 가짜 임베딩 벡터
    fake_context_doc = Document(
        page_content="검색된 가짜 컨텍스트입니다.",
        metadata={"title": "가짜 문서"}
    )
    final_rag_answer = "Ollama가 생성한 최종 답변입니다."

    # Mock 1: Ollama Client 모방 (두 번 호출됨)
    mock_ollama_client = MagicMock()
    mock_hyde_response = {'message': {'content': fake_hyde_answer}}
    mock_rag_response = {'message': {'content': final_rag_answer}}

    # .chat()가 호출될 때마다 순서대로 다른 값을 반환하도록 설정
    mock_ollama_client.chat.side_effect = [
        mock_hyde_response,
        mock_rag_response
    ]
    # 'run_hyde_search' 파일에서 'ollama.Client'를 찾아서 모방
    mocker.patch("run_hyde_search.ollama.Client", return_value=mock_ollama_client)

    # Mock 2: Elasticsearch Embedding 함수 모방
    mock_embed_func = mocker.patch(
        "run_hyde_search.elasticsearch_manager.embed_text",
        return_value=fake_vector
    )

    # Mock 3: Elasticsearch k-NN 검색 함수 모방
    mock_search_func = mocker.patch(
        "run_hyde_search.search_pipeline.search_es_knn",
        return_value=[fake_context_doc]
    )

    # --- 2. Act (테스트 대상 함수 실행) ---
    response = run_hyde_search.run_hyde_pipeline(
        test_question,
        mock_es_client,
        mock_tokenizer,
        mock_model
    )

    # --- 3. Assert (흐름 검증) ---

    # 3-1. 최종 반환값이 Ollama의 두 번째 응답과 일치하는가?
    assert response == final_rag_answer

    # 3-2. Ollama 클라이언트가 총 2번 호출되었는가?
    assert mock_ollama_client.chat.call_count == 2

    # 3-3. 첫 번째 (HyDE) 호출이 '원본 질문'을 포함하여 호출되었는가?
    first_call_args = mock_ollama_client.chat.call_args_list[0]
    assert test_question in first_call_args.kwargs['messages'][0]['content']

    # 3-4. 임베딩 함수가 'HyDE의 가짜 답변'으로 호출되었는가?
    mock_embed_func.assert_called_once_with(
        fake_hyde_answer,
        mock_tokenizer,
        mock_model
    )

    # 3-5. ES 검색 함수가 '임베딩된 가짜 벡터'로 호출되었는가?
    mock_search_func.assert_called_once_with(
        es_client=mock_es_client,
        query_vector=fake_vector,
        k=3  # 코드에 하드코딩된 k=3 값과 일치하는지 확인
    )

    # 3-6. 두 번째 (RAG) 호출이 '원본 질문'과 '검색된 컨텍스트'를 포함했는가?
    second_call_args = mock_ollama_client.chat.call_args_list[1]
    rag_prompt_content = second_call_args.kwargs['messages'][0]['content']
    assert test_question in rag_prompt_content
    assert fake_context_doc.page_content in rag_prompt_content