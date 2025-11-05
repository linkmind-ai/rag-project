# config/__init__.py

import os
from dotenv import load_dotenv

# --- 1. .env 파일 로드 ---
# 프로젝트 루트에 있는 .env 파일을 찾아 로드합니다.
load_dotenv()

# --- 2. .env에서 변수 로드 (Secrets, Hosts, Paths) ---
NOTION_TOKEN = os.environ.get("NOTION_TOKEN")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_ID = os.environ.get("ES_ID")
ES_HOST = os.environ.get("ES_HOST")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST")

JSON_SAVE_PATH = os.environ.get("JSON_SAVE_PATH")
TXT_SAVE_PATH = os.environ.get("TXT_SAVE_PATH")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH")

# --- 3. 필수 값 검증 ---
if not NOTION_TOKEN:
    raise ValueError("❌ 'NOTION_TOKEN'을 .env 파일에서 찾을 수 없습니다.")
if not ES_API_KEY or not ES_ID:
    raise ValueError("❌ 'ES_API_KEY' 또는 'ES_ID'를 .env 파일에서 찾을 수 없습니다.")
if not ES_HOST:
    raise ValueError("❌ 'ES_HOST'를 .env 파일에서 찾을 수 없습니다.")

# --- 4. 코드에 하드코딩된 설정값 (Constants) ---
NOTION_VERSION = "2022-06-28"
ES_INDEX_NAME = "vector-test-index"
ES_EMBEDDING_DIMS = 384

# --- 5. 모델 이름 ---
KEYBERT_MODEL = "jhgan/ko-sroberta-multitask"
ES_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_EMBEDDING_MODEL = "jhgan/ko-sbert-sts"
RERANKER_MODEL = "BAAI/bge-reranker-base"
OLLAMA_MODEL = "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M"

# Pytest
TEST_NOTION_PAGE_ID = "1dd124ddd3138059983afff89ceb5ea4"

print("✅ [Config] 설정이 .env 파일과 config 패키지로부터 로드되었습니다.")