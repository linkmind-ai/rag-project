# src/storage/elastic_store.py

from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.schema import Document
# [수정] config 임포트 방식 변경
from config import ES_EMBEDDING_MODEL  # config는 top-level 패키지


class ElasticStore:
    """Elasticsearch 연결, 임베딩, 인덱싱, 검색을 모두 처리하는 클래스"""

    def __init__(self, host, api_id, api_key, index_name, embedding_model_name=ES_EMBEDDING_MODEL):
        self.index_name = index_name

        try:
            self.client = Elasticsearch(
                host,
                api_key=(api_id, api_key),
                verify_certs=False,
                ssl_show_warn=False
            )
            if not self.client.ping():
                raise ValueError("Elasticsearch 서버에 연결할 수 없습니다.")
            print(f"✅ Elasticsearch 연결 성공 (Host: {host})")
        except Exception as e:
            print(f"❌ Elasticsearch 연결 오류: {e}")
            raise

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            self.model = AutoModel.from_pretrained(embedding_model_name)
            print(f"✅ 임베딩 모델 로드 성공: {embedding_model_name}")
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 오류: {e}")
            raise

    def _embed_text(self, text: str) -> list[float]:
        """텍스트를 임베딩 벡터로 변환합니다."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.numpy().tolist()

    def create_index_if_not_exists(self, dims: int):
        """설정된 인덱스 이름으로 인덱스를 생성합니다."""
        if not self.client.indices.exists(index=self.index_name):
            try:
                self.client.indices.create(
                    index=self.index_name,
                    mappings={
                        "properties": {
                            "content": {"type": "text"},
                            "metadata": {"type": "object"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": dims,
                                "index": True,
                                "similarity": "cosine",
                            }
                        }
                    }
                )
                print(f"✅ 인덱스 '{self.index_name}' 생성 완료.")
            except Exception as e:
                print(f"❌ 인덱스 생성 오류: {e}")
        else:
            print(f"ℹ️ 인덱스 '{self.index_name}'가 이미 존재합니다.")

    def index_documents(self, documents: list[Document]):
        """문서 목록을 임베딩하여 Elasticsearch에 저장합니다."""
        print(f"'{self.index_name}' 인덱스에 문서 임베딩 및 인덱싱 시작...")
        for i, doc in enumerate(documents):
            try:
                body = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": self._embed_text(doc.page_content),
                }
                self.client.index(index=self.index_name, id=f"chunk-{i}", document=body)
            except Exception as e:
                print(f"❌ 문서 {i} 인덱싱 오류: {e}")

        self.client.indices.refresh(index=self.index_name)
        print(f"✅ 총 {len(documents)}개의 문서가 Elasticsearch에 저장되었습니다.")

    def search_knn(self, query_text: str, k=3) -> list[Document]:
        """텍스트 쿼리를 벡터로 변환하여 k-NN 검색을 실행합니다."""
        query_vector = self._embed_text(query_text)
        knn_query = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": 100
        }

        try:
            response = self.client.search(
                index=self.index_name,
                knn=knn_query,
                _source_excludes=["embedding"]
            )
            contexts_docs = []
            for hit in response['hits']['hits']:
                doc = Document(
                    page_content=hit['_source'].get('content', ''),
                    metadata=hit['_source'].get('metadata', {})
                )
                contexts_docs.append(doc)
            return contexts_docs
        except Exception as e:
            print(f"❌ Elasticsearch k-NN 검색 오류: {e}")
            return []