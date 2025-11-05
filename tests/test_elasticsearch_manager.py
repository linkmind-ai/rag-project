# tests/test_elasticsearch_manager.py

import pytest
from unittest.mock import MagicMock, ANY
from langchain.schema import Document
import elasticsearch_manager


@pytest.fixture
def mock_es_client(mocker):
    """가짜 Elasticsearch 클라이언트를 생성합니다."""
    mock_es = MagicMock()
    mock_es.ping.return_value = True
    mock_es.indices.exists.return_value = False
    mock_es.count.return_value = {"count": 1}
    mock_es.search.return_value = {"hits": {"hits": [
        {"_id": "chunk-0", "_source": {"content": "테스트", "metadata": {}}}
    ]}}

    # 📌 [수정 1]
    # 'elasticsearch.Elasticsearch' (원본) 대신
    # 'elasticsearch_manager.Elasticsearch' (실제 사용되는 곳)를 패치해야 합니다.
    mocker.patch("elasticsearch_manager.Elasticsearch", return_value=mock_es)

    return mock_es


@pytest.fixture
def mock_embedding_model(mocker):
    """가짜 임베딩 모델을 생성합니다."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mocker.patch(
        "elasticsearch_manager.embed_text",
        return_value=[0.1] * 384
    )

    mocker.patch("elasticsearch_manager.get_embedding_model", return_value=(mock_tokenizer, mock_model))
    return mock_tokenizer, mock_model


def test_get_es_client_success(mock_es_client):
    """ES 클라이언트 연결 성공 테스트"""
    es = elasticsearch_manager.get_es_client()
    assert es is not None

    # 📌 [수정 1의 결과]
    # 이제 'es'는 진짜 클라이언트가 아닌 'mock_es_client' (MagicMock)이므로,
    # 'es.ping'은 mock 객체이며 '.assert_called_once()'를 사용할 수 있습니다.
    # 또한, 이 테스트는 더 이상 실제 네트워크에 접속하지 않으므로,
    # 'SecurityWarning'과 'InsecureRequestWarning' 경고도 사라집니다.
    es.ping.assert_called_once()


def test_create_es_index(mock_es_client):
    """ES 인덱스 생성 테스트"""
    mock_es_client.indices.exists.return_value = False

    elasticsearch_manager.create_es_index(
        mock_es_client,
        index_name="test-idx",
        vec_dims=384
    )

    mock_es_client.indices.create.assert_called_with(
        index="test-idx",
        mappings=ANY
    )


def test_index_documents(mock_es_client, mock_embedding_model):
    """(이 테스트는 이미 PASSED) 문서 1개가 정상적으로 인덱싱되는지 테스트"""
    tokenizer, model = mock_embedding_model
    dummy_doc = Document(page_content="테스트 문서", metadata={})

    elasticsearch_manager.index_documents(
        mock_es_client, [dummy_doc], tokenizer, model, index_name="test-idx"
    )

    mock_es_client.index.assert_called_once_with(
        index="test-idx",
        id="chunk-0",
        document={
            "content": "테스트 문서",
            "metadata": {},
            "embedding": [0.1] * 384
        }
    )