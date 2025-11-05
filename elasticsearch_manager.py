# elasticsearch_manager.py

from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
import config


def get_es_client():
    """Elasticsearch 클라이언트를 반환합니다."""
    try:
        es = Elasticsearch(
            config.ES_HOST,
            api_key=(config.ES_ID, config.ES_API_KEY),
            # ⚠️ 'es.nabee.ai.kr'이 자체 서명 인증서를 사용할 경우
            # verify_certs=False 또는 ca_certs 경로 지정이 필요할 수 있습니다.
            verify_certs=False, # 노트북 코드에 맞춰 우선 False로 설정
            ssl_show_warn = False
        )
        if not es.ping():
            raise ValueError("Elasticsearch 서버에 연결할 수 없습니다. config.py를 확인하세요.")
        print("✅ Elasticsearch 연결 성공!")
        return es
    except Exception as e:
        print(f"❌ Elasticsearch 연결 오류: {e}")
        return None


def get_embedding_model():
    """HuggingFace 임베딩 모델과 토크나이저를 로드합니다."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.ES_EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(config.ES_EMBEDDING_MODEL)
        print(f"✅ 임베딩 모델 로드 성공: {config.ES_EMBEDDING_MODEL}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 오류: {e}")
        return None, None


def embed_text(text, tokenizer, model):
    """텍스트를 임베딩 벡터로 변환합니다."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy().tolist()


def create_es_index(es_client, index_name, vec_dims):
    """Elasticsearch에 벡터 인덱스를 생성합니다."""
    if not es_client.indices.exists(index=index_name):
        try:
            es_client.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        "content": {"type": "text"},
                        "metadata": {"type": "object"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": vec_dims,
                            "index": True,
                            "similarity": "cosine",
                        }
                    }
                }
            )
            print(f"✅ 인덱스 '{index_name}' 생성 완료.")
        except Exception as e:
            print(f"❌ 인덱스 생성 오류: {e}")
    else:
        print(f"ℹ️ 인덱스 '{index_name}'가 이미 존재합니다.")


def delete_all_docs(es_client, index_name):
    """인덱스 내 모든 문서를 삭제합니다."""
    try:
        es_client.delete_by_query(
            index=index_name,
            body={"query": {"match_all": {}}},
            refresh=True
        )
        print(f"🧩 인덱스 '{index_name}' 내 모든 문서가 삭제되었습니다.")
    except Exception as e:
        print(f"❌ 문서 삭제 오류: {e}")


def index_documents(es_client, documents, tokenizer, model, index_name):
    """문서 목록을 임베딩하여 Elasticsearch에 저장합니다."""
    print(f"'{index_name}' 인덱스에 문서 임베딩 및 인덱싱 시작...")
    for i, doc in enumerate(documents):
        content = doc.page_content
        metadata = doc.metadata

        try:
            body = {
                "content": content,
                "metadata": metadata,
                "embedding": embed_text(content, tokenizer, model),
            }
            es_client.index(index=index_name, id=f"chunk-{i}", document=body)
        except Exception as e:
            print(f"❌ 문서 {i} 인덱싱 오류: {e}")

    es_client.indices.refresh(index=index_name)
    print(f"✅ 총 {len(documents)}개의 문서가 Elasticsearch에 저장되었습니다.")


def search_all_docs(es_client, index_name):
    """Elasticsearch에서 모든 문서를 검색합니다 (임베딩 제외)."""
    try:
        count = es_client.count(index=index_name)["count"]
        if count == 0:
            print(f"ℹ️ '{index_name}' 인덱스에 문서가 없습니다.")
            return

        res = es_client.search(
            index=index_name,
            query={"match_all": {}},
            size=count,
            _source_excludes=["embedding"]
        )
        print(f"--- '{index_name}' 인덱스 내 문서 조회 (총 {len(res['hits']['hits'])}개) ---")
        for i, hit in enumerate(res["hits"]["hits"]):
            print('-' * 20)
            print(f"{i + 1}번째 content (ID: {hit['_id']})")
            print(hit['_source'])
    except Exception as e:
        print(f"❌ 문서 검색 오류: {e}")