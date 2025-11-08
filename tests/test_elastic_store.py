# tests/test_elastic_store.py

import pytest
from unittest.mock import MagicMock, ANY
from langchain.schema import Document
from src.storage.elastic_store import ElasticStore


@pytest.fixture
def mock_es_dependencies(mocker):
    """ElasticStore의 __init__에서 사용하는 모든 외부 의존성을 모방합니다."""

    # 1. Elasticsearch 클라이언트 모방
    mock_es = MagicMock()
    mock_es.ping.return_value = True
    mock_es.indices.exists.return_value = False
    mock_es.search.return_value = {
        'hits': {'hits': [{'_source': {'content': 'doc1', 'metadata': {}}}]}
    }
    mocker.patch("src.storage.elastic_store.Elasticsearch", return_value=mock_es)

    # 2. Transformers 모델/토크나이저 모방
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mocker.patch("src.storage.elastic_store.AutoTokenizer.from_pretrained", return_value=mock_tokenizer)
    mocker.patch("src.storage.elastic_store.AutoModel.from_pretrained", return_value=mock_model)

    # 3. _embed_text 메서드 모방 (벡터값 반환)
    mocker.patch(
        "src.storage.elastic_store.ElasticStore._embed_text",
        return_value=[0.1] * 384
    )

    return mock_es, mock_tokenizer, mock_model


def test_elastic_store_init(mock_es_dependencies):
    """ElasticStore 초기화가 ES Ping과 모델 로드를 호출하는지 테스트합니다."""
    mock_es, mock_tokenizer, mock_model = mock_es_dependencies

    store = ElasticStore("host", "id", "key", "index")

    assert store.client is not None
    store.client.ping.assert_called_once()
    assert store.tokenizer is not None
    assert store.model is not None


def test_create_index_if_not_exists(mock_es_dependencies):
    """인덱스 생성 로직 테스트"""
    mock_es, _, _ = mock_es_dependencies
    mock_es.indices.exists.return_value = False  # 인덱스가 없다고 가정

    store = ElasticStore("host", "id", "key", "test-index")
    store.create_index_if_not_exists(dims=384)

    mock_es.indices.create.assert_called_with(
        index="test-index",
        mappings=ANY
    )


def test_index_documents(mock_es_dependencies):
    """문서 색인 로직 테스트"""
    mock_es, _, _ = mock_es_dependencies
    dummy_doc = Document(page_content="테스트 문서", metadata={})

    store = ElasticStore("host", "id", "key", "test-index")
    store.index_documents([dummy_doc])

    # _embed_text가 호출되고, 그 결과로 client.index가 호출되었는지 검증
    store._embed_text.assert_called_with("테스트 문서")
    mock_es.index.assert_called_once_with(
        index="test-index",
        id="chunk-0",
        document={
            "content": "테스트 문서",
            "metadata": {},
            "embedding": [0.1] * 384
        }
    )


def test_search_knn(mock_es_dependencies):
    """k-NN 검색 로직 테스트"""
    mock_es, _, _ = mock_es_dependencies

    store = ElasticStore("host", "id", "key", "test-index")
    results = store.search_knn("테스트 쿼리", k=1)

    # 1. _embed_text가 쿼리로 호출되었는지 확인
    store._embed_text.assert_called_with("테스트 쿼리")

    # 2. client.search가 올바른 k-NN 쿼리로 호출되었는지 확인
    mock_es.search.assert_called_with(
        index="test-index",
        knn={
            "field": "embedding",
            "query_vector": [0.1] * 384,  # _embed_text의 모방된 반환값
            "k": 1,
            "num_candidates": 100
        },
        _source_excludes=["embedding"]
    )

    # 3. 결과가 Document 객체로 변환되었는지 확인
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].page_content == "doc1"