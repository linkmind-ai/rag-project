# download_models.py

import config
# from kiwipiepy import Kiwi # 제거
from konlpy.tag import Okt  # 추가
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import FlagReranker

print("--- 1. Caching Konlpy Okt model ---")
try:
    # Okt는 초기화 시 Java(JPype)를 로드하고 필요한 파일을 캐시합니다.
    Okt()
    print("✅ Konlpy Okt model OK.")
except Exception as e:
    print(f"❌ Konlpy Okt download/load failed: {e}")
    print("   (JDK가 설치되어 있는지 확인하세요.)")

print(f"\n--- 2. Caching KeyBERT model ({config.KEYBERT_MODEL}) ---")
try:
    SentenceTransformer(config.KEYBERT_MODEL)
    print("✅ KeyBERT model OK.")
except Exception as e:
    print(f"❌ KeyBERT model download/load failed: {e}")

print(f"\n--- 3. Caching ES Embedding model ({config.ES_EMBEDDING_MODEL}) ---")
try:
    AutoTokenizer.from_pretrained(config.ES_EMBEDDING_MODEL)
    AutoModel.from_pretrained(config.ES_EMBEDDING_MODEL)
    print("✅ ES Embedding model (transformers) OK.")
except Exception as e:
    print(f"❌ ES Embedding model download/load failed: {e}")

print(f"\n--- 4. Caching Reranker model ({config.RERANKER_MODEL}) ---")
try:
    FlagReranker(config.RERANKER_MODEL, use_fp16=True)
    print("✅ Reranker model OK.")
except Exception as e:
    print(f"❌ Reranker model download/load failed: {e}")

print("\n--- 🏁 All models pre-downloaded/cached. ---")