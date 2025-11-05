# search_pipeline.py

import config
from langchain.schema import Document


def search_es_knn(es_client, query_vector, index_name=config.ES_INDEX_NAME, k=3):
    """
    Elasticsearch에 k-NN 벡터 검색을 실행합니다.
    HyDE 검색 결과(문서 리스트)를 반환합니다.
    """

    knn_query = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": 100  # k보다 큰 값을 설정하여 검색 품질 향상
    }

    try:
        response = es_client.search(
            index=index_name,
            knn=knn_query,
            _source_excludes=["embedding"]  # 결과에서 무거운 임베딩 벡터는 제외
        )

        # Langchain Document 형식으로 변환하여 반환
        contexts_docs = []
        for hit in response['hits']['hits']:
            content = hit['_source'].get('content', '')
            metadata = hit['_source'].get('metadata', {})
            doc = Document(page_content=content, metadata=metadata)
            contexts_docs.append(doc)

        return contexts_docs

    except Exception as e:
        print(f"❌ Elasticsearch k-NN 검색 오류: {e}")
        return []