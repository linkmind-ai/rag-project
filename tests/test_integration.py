# tests/test_integration.py

import pytest
import config
from src.clients.notion_client import NotionClient
from src.storage.elastic_store import ElasticStore
from src.clients.ollama_client import OllamaClient
from src.services.rag_agent import RagAgent
from langchain.schema import Document
import time

# 이 파일의 모든 테스트는 'integration' 마커를 가짐
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_store():
    """[Integration] 실제 ES 클라이언트와 임베딩 모델을 로드합니다."""
    store = ElasticStore(
        host=config.ES_HOST,
        api_id=config.ES_ID,
        api_key=config.ES_API_KEY,
        index_name=config.ES_INDEX_NAME,  # config의 기본 인덱스 사용
        embedding_model_name=config.ES_EMBEDDING_MODEL
    )
    # __init__에서 ping/load가 이미 완료됨
    return store


@pytest.fixture(scope="module")
def real_rag_agent(real_store):
    """[Integration] 실제 RAG 에이전트를 로드합니다."""
    ollama = OllamaClient(
        host=config.OLLAMA_HOST,
        model=config.OLLAMA_MODEL
    )
    agent = RagAgent(vector_store=real_store, llm_client=ollama)
    return agent


# --- 테스트 함수 ---

def test_notion_api_connection():
    """1. (통합) 실제 Notion API 연결 테스트"""
    print(f"Notion API 테스트 (Page ID: {config.TEST_NOTION_PAGE_ID})")
    assert config.TEST_NOTION_PAGE_ID, "config.py에 TEST_NOTION_PAGE_ID를 설정하세요"

    client = NotionClient(token=config.NOTION_TOKEN, version=config.NOTION_VERSION)
    page_data = client.fetch_all_blocks(config.TEST_NOTION_PAGE_ID)

    assert page_data is not None
    assert len(page_data) > 0
    print("✅ Notion API 연결 및 데이터 수신 성공")


def test_elasticsearch_api_connection(real_store):
    """2. (통합) 실제 Elasticsearch API 연결 테스트"""
    print(f"Elasticsearch API 테스트 (Host: {config.ES_HOST})")
    # real_store fixture가 로드되는 순간 이 테스트는 통과한 것임.
    assert real_store.client is not None
    info = real_store.client.info()
    assert info.get('version', {}).get('number', '').startswith("9.1.5")
    print("✅ Elasticsearch API 연결 성공")


def test_elasticsearch_knn_search_cycle(real_store):
    """3. (통합) 실제 ES 서버에 [색인 -> k-NN 검색 -> 삭제] 문서 단위 사이클 테스트"""

    TEST_INDEX = config.ES_INDEX_NAME
    TEST_DOC_ID = "pytest-integration-test-doc"

    print(f"Elasticsearch k-NN R/W 테스트 (Index: {TEST_INDEX}, DocID: {TEST_DOC_ID})")

    try:
        # 1. (Test) 테스트 문서 색인
        doc_content = "통합 테스트용 벡터 검색 문서입니다."
        doc_vector = real_store._embed_text(doc_content)  # 내부 메서드 사용

        doc_body = {
            "content": doc_content,
            "metadata": {"source": "pytest-knn"},
            "embedding": doc_vector
        }

        real_store.client.index(
            index=TEST_INDEX,
            id=TEST_DOC_ID,
            document=doc_body,
            refresh=True
        )
        print("  [1/3] 테스트 문서 색인 완료")

        # 2. (Test) k-NN 검색 테스트
        # ElasticStore의 search_knn 메서드를 직접 테스트
        search_results = real_store.search_knn("벡터 검색 테스트", k=1)

        assert len(search_results) >= 1
        found = any(doc.metadata.get("source") == "pytest-knn" for doc in search_results)
        assert found, "k-NN 검색 결과에서 테스트 문서를 찾지 못했습니다."
        print("  [2/3] k-NN 검색 및 검증 완료")

    finally:
        # 3. (Teardown) 테스트 문서 삭제
        try:
            real_store.client.options(ignore_status=[400, 404]).delete(
                index=TEST_INDEX,
                id=TEST_DOC_ID
            )
            print(f"  [3/3] 테스트 문서 '{TEST_DOC_ID}' 삭제 완료")
        except Exception as e:
            print(f"⚠️ 테스트 문서 '{TEST_DOC_ID}' 삭제 실패: {e}")

    print("✅ Elasticsearch k-NN R/W 사이클 테스트 성공")


def test_full_rag_pipeline(real_rag_agent):
    """4. (통합) 실제 RAG 파이프라인(Ollama -> ES -> Ollama) 전체 실행 테스트"""

    print(f"\nFull RAG Pipeline (Ollama -> ES -> Ollama) 테스트")

    # ⚠️ `main_etl.py`을 통해 실제 색인된 데이터와 관련된 질문이어야 함
    question = "AI의 가치 정렬 문제는 구체적으로 어떤 우려를 말하나요?"

    # 1. Act (실제 파이프라인 실행)
    final_response = real_rag_agent.query(question)

    # 2. Assert (검증)
    assert final_response is not None, "Ollama RAG 파이프라인이 응답을 반환하지 않았습니다."
    assert isinstance(final_response, str)
    assert len(final_response) > 0, "Ollama가 빈 문자열을 반환했습니다."

    print(f"✅ Full RAG Pipeline 성공. 응답 (앞 50자): {final_response[:50]}...")