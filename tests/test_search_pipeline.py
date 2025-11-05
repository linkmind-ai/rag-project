# tests/test_search_pipeline.py

import pytest
from unittest.mock import MagicMock
from langchain.schema import Document
import search_pipeline


@pytest.fixture
def mock_es_client(mocker):
    """가짜 Elasticsearch 클라이언트를 생성합니다."""
    mock_es = MagicMock()

    # k-NN 검색에 대한 가짜 응답
    mock_response = {
        'hits': {
            'hits': [
                {
                    '_id': 'chunk-0',
                    '_source': {
                        'content': '이것은 k-NN 검색 결과입니다.',
                        'metadata': {'title': '벡터 문서'}
                    }
                }
            ]
        }
    }
    mock_es.search.return_value = mock_response
    return mock_es


def test_search_es_knn(mock_es_client):
    """
    search_es_knn 함수가 올바른 k-NN 쿼리를 생성하여
    es_client.search를 호출하는지 테스트합니다.
    """
    test_vector = [0.1, 0.2, 0.3]
    test_k = 1

    # 함수 실행
    results = search_pipeline.search_es_knn(
        es_client=mock_es_client,
        query_vector=test_vector,
        index_name="test-index",
        k=test_k
    )

    # 1. es_client.search가 호출되었는지 확인
    mock_es_client.search.assert_called_once()

    # 2. 호출 시 사용된 'knn' 인자 검증
    # (kwargs['knn']를 통해 실제 전달된 knn 쿼리 객체를 가져옴)
    args, kwargs = mock_es_client.search.call_args
    assert "knn" in kwargs
    knn_query = kwargs["knn"]

    assert knn_query["field"] == "embedding"
    assert knn_query["k"] == test_k
    assert knn_query["query_vector"] == test_vector
    assert "num_candidates" in knn_query

    # 3. 반환값이 Langchain Document 형식인지 확인
    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert results[0].page_content == '이것은 k-NN 검색 결과입니다.'
    assert results[0].metadata['title'] == '벡터 문서'