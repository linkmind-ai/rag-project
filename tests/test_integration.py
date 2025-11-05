# tests/test_integration.py

import pytest
import config  # config/ __init__.py 로드
import notion_loader
import elasticsearch_manager
import search_pipeline
import run_hyde_search  # 👈 [추가] RAG 파이프라인 함수를 임포트
from langchain.schema import Document
import time

# 이 파일의 모든 테스트는 'integration' 마커를 가짐
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def es_client():
    """테스트 전체에서 사용할 실제 ES 클라이언트 (1회만 생성)"""
    client = elasticsearch_manager.get_es_client()
    assert client is not None, "Elasticsearch 연결 실패. config.py 또는 서버 상태 확인"
    return client


@pytest.fixture(scope="module")
def embedding_models():
    """테스트에서 사용할 실제 임베딩 모델 (1회만 로드)"""
    tokenizer, model = elasticsearch_manager.get_embedding_model()
    assert tokenizer is not None and model is not None, "임베딩 모델 로드 실패"
    return tokenizer, model


# --- 1. Notion API 테스트 ---
def test_notion_api_connection():
    """(통합) 실제 Notion API 연결 및 데이터 수신 테스트"""
    print(f"Notion API 테스트 (Page ID: {config.TEST_NOTION_PAGE_ID})")
    assert config.TEST_NOTION_PAGE_ID, "config.py에 TEST_NOTION_PAGE_ID를 설정하세요"

    page_data = notion_loader.get_page_content(config.TEST_NOTION_PAGE_ID)
    assert page_data is not None, "Notion 페이지 속성을 가져오지 못했습니다. (토큰 또는 ID 확인)"

    blocks = notion_loader.fetch_all_blocks(config.TEST_NOTION_PAGE_ID)
    assert blocks is not None, "Notion 블록을 가져오지 못했습니다."
    print("✅ Notion API 연결 및 데이터 수신 성공")


# --- 2. Elasticsearch 연결 테스트 ---
def test_elasticsearch_api_connection(es_client):
    """(통합) 실제 Elasticsearch API 연결 테스트"""
    print(f"Elasticsearch API 테스트 (Host: {config.ES_HOST})")
    info = es_client.info()
    assert "cluster_name" in info
    assert info.get('version', {}).get('number', '').startswith("9.1.5")
    print(f"✅ Elasticsearch API 연결 성공 (Cluster: {info['cluster_name']})")


# --- 3. Elasticsearch R/W 및 k-NN 검색 테스트 ---
def test_elasticsearch_knn_search_cycle(es_client, embedding_models):
    """(통합) 실제 ES 서버에 [색인 -> k-NN 검색 -> 삭제] 문서 단위 사이클 테스트"""

    TEST_INDEX = config.ES_INDEX_NAME
    TEST_DOC_ID = "pytest-integration-test-doc"

    tokenizer, model = embedding_models

    print(f"Elasticsearch k-NN R/W 테스트 (Index: {TEST_INDEX}, DocID: {TEST_DOC_ID})")

    try:
        # 1. (Test) 테스트 문서 색인
        doc_content = "통합 테스트용 벡터 검색 문서입니다."
        doc_vector = elasticsearch_manager.embed_text(doc_content, tokenizer, model)

        doc_body = {
            "content": doc_content,
            "metadata": {"source": "pytest-knn"},
            "embedding": doc_vector
        }

        es_client.index(
            index=TEST_INDEX,
            id=TEST_DOC_ID,
            document=doc_body,
            refresh=True
        )
        print("  [1/3] 테스트 문서 색인 완료")

        # 2. (Test) k-NN 검색 테스트
        query_vector = elasticsearch_manager.embed_text("벡터 검색 테스트", tokenizer, model)

        search_results = search_pipeline.search_es_knn(
            es_client, query_vector, TEST_INDEX, k=1
        )

        assert len(search_results) >= 1
        found = any(doc.metadata.get("source") == "pytest-knn" for doc in search_results)
        assert found, "k-NN 검색 결과에서 테스트 문서를 찾지 못했습니다."
        print("  [2/3] k-NN 검색 및 검증 완료")

    finally:
        # 3. (Teardown) 테스트 문서 삭제
        try:
            es_client.options(ignore_status=[400, 404]).delete(
                index=TEST_INDEX,
                id=TEST_DOC_ID
            )
            print(f"  [3/3] 테스트 문서 '{TEST_DOC_ID}' 삭제 완료")
        except Exception as e:
            print(f"⚠️ 테스트 문서 '{TEST_DOC_ID}' 삭제 실패: {e}")

    print("✅ Elasticsearch k-NN R/W 사이클 테스트 성공")


# --- 4. 🚀 [신규 추가] 전체 RAG 파이프라인 통합 테스트 ---
def test_full_rag_pipeline(es_client, embedding_models):
    """
    (통합) 실제 RAG 파이프라인(Ollama -> ES -> Ollama) 전체 실행 테스트
    config.py의 OLLAMA_HOST가 유효해야 하며,
    config.ES_INDEX_NAME에 'main_workflow.py'를 통해 데이터가 색인되어 있어야 합니다.
    """

    print(f"\nFull RAG Pipeline (Ollama -> ES -> Ollama) 테스트 (Host: {config.OLLAMA_HOST})")

    tokenizer, model = embedding_models

    # ⚠️ `main_workflow.py`를 통해 실제 색인된 데이터와 관련된 질문이어야 합니다.
    question = "AI의 가치 정렬 문제는 구체적으로 어떤 우려를 말하나요?"

    # --- 1. Act (실제 파이프라인 실행) ---
    final_response = run_hyde_search.run_hyde_pipeline(
        question, es_client, tokenizer, model
    )

    # --- 2. Assert (검증) ---
    assert final_response is not None, "Ollama RAG 파이프라인이 응답을 반환하지 않았습니다."
    assert isinstance(final_response, str), "Ollama 응답이 문자열(str)이 아닙니다."
    assert len(final_response) > 0, "Ollama가 빈 문자열을 반환했습니다."

    print(f"✅ Full RAG Pipeline 성공. 응답 (앞 50자): {final_response[:50]}...")