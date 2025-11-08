

# Notion RAG 파이프라인 (Elasticsearch & Ollama)

### 📖 개요

이 프로젝트는 Notion 문서를 수집·정제하고, Elasticsearch를 벡터 DB로 사용하여 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하는 것을 목표로 합니다.

**핵심 기능:**

  * **ETL:** Notion API로 문서를 수집, `Okt`와 `KeyBERT`로 한국어 텍스트를 처리(청킹, 키워드 추출)하여 `Elasticsearch`에 벡터로 색인합니다.
  * **RAG:** `Ollama`를 사용하여 사용자 질문을 HyDE(가상 문서)로 확장하고, `Elasticsearch` (k-NN)에서 관련 문서를 검색한 뒤, `Ollama`가 검색된 컨텍스트를 기반으로 최종 답변을 생성합니다.

-----

### 🚀 1. 빠른 시작: 설치 및 환경 설정

새로운 팀원이 프로젝트를 실행하기 위한 단계입니다.

#### 1.1. 프로젝트 복제

```bash
git clone [YOUR_REPOSITORY_URL]
cd rag-project
```

#### 1.2. `.env` 파일 생성 (필수)

프로젝트 루트에 `.env` 파일을 생성하고, 아래 `.env.example` 내용을 복사한 뒤, 팀 리더에게 공유받은 실제 비밀 키를 채워 넣습니다.

**`.env.example` (이 양식을 `.env` 파일로 복사):**

```ini
# --- Secrets (팀 공유 필요) ---
NOTION_TOKEN="ntn_YOUR_NOTION_INTEGRATION_TOKEN"
ES_API_KEY="YOUR_ELASTICSEARCH_API_KEY"
ES_ID="YOUR_ELASTICSEARCH_API_KEY_ID"

# --- Environment-Specific Hosts (팀 공유) ---
ES_HOST="https://your-es-instance.com"
OLLAMA_HOST="https://your-ollama-instance.com"

# --- User-Specific File Paths (개인 경로) ---
# (경로에 한글/공백이 없도록 주의)
JSON_SAVE_PATH="/Users/your_name/projects/rag-project/data/notion_content.json"
TXT_SAVE_PATH="/Users/your_name/projects/rag-project/data/notion_content.txt"

# --- Test-Specific (팀 공유) ---
# 통합 테스트(pytest -m integration)에 사용할 Notion 페이지 ID
TEST_NOTION_PAGE_ID="YOUR_TEST_NOTION_PAGE_ID"
```

> **보안 ⚠️:** `.env` 파일은 `.gitignore`에 반드시 포함되어야 하며, **절대 Git에 커밋하면 안 됩니다.**

#### 1.3. 가상 환경 및 라이브러리 설치

Python 3.11+ 환경을 권장합니다.

```bash
# 1. 가상 환경 생성 및 활성화
python3.11 -m venv .venv
source .venv/bin/activate

# 2. 요구사항 설치
pip install -r requirements.txt
```

#### 1.4. (중요) AI 모델 캐시(Cache) 생성

`pytest` 또는 `api` 실행 시 C++/Java 라이브러리 충돌을 방지하기 위해, **최초 1회** 모델을 미리 로드하는 스크립트를 실행해야 합니다.

```bash
# 📣 [Intel Mac 사용자 경고]
# C++ OpenMP 라이브러리 충돌(Fatal Python error: Aborted)을 방지하려면
# 아래와 같이 KMP_DUPLICATE_LIB_OK=TRUE 환경 변수를 반드시 붙여야 합니다.
# (M2/M3 Mac 사용자는 이 변수 없이 실행해도 됩니다.)
#
KMP_DUPLICATE_LIB_OK=TRUE python -m scripts.download_models
```

*(이 스크립트는 모델을 "다운로드"하는 것이 아니라, 이미 `pip install`로 설치된 모델을 메모리에 "미리 로드"하여 초기화 충돌을 방지하는 워밍업 작업입니다.)*

-----

### \#\# 🏃‍♂️ 2. 프로젝트 실행

이 프로젝트는 2개의 메인 애플리케이션(`jobs`, `api`)으로 구성됩니다.

#### 2.1. (Offline) ETL 파이프라인 실행

Notion에서 데이터를 가져와 Elasticsearch에 색인합니다. (데이터가 변경될 때마다 실행 필요)

```bash
# (venv)
# 📣 (Intel Mac) KMP_DUPLICATE_LIB_OK=TRUE 를 붙여 실행하세요.
KMP_DUPLICATE_LIB_OK=TRUE python -m jobs.batch.run_notion_etl
```

실행하면 터미널에 Notion 페이지 ID를 입력하라는 메시지가 나타납니다.

#### 2.2. (Online) RAG API 서버 실행

사용자 질문에 답변하는 FastAPI 서버를 실행합니다.

```bash
# (venv)
# 📣 (Intel Mac) KMP_DUPLICATE_LIB_OK=TRUE 를 붙여 실행하세요.
KMP_DUPLICATE_LIB_OK=TRUE uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

서버가 `http://localhost:8000`에서 실행됩니다.

#### 2.3. API 테스트

서버가 실행된 상태에서, **새 터미널**을 열어 `curl`로 `POST` 요청을 보내 테스트합니다.

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "AI의 가치 정렬 문제는 무엇인가요?"}'
```

**성공 응답 예시:**

```json
{
  "answer": "AI의 가치 정렬 문제는 AI가 인간의 의도나 가치와 어긋나는..."
}
```

-----

### \#\# 🏛️ 3. 아키텍처: "Separated Layout"

이 프로젝트는 Python 표준 "Separated Layout" 구조를 따릅니다. 이는 \*\*"재사용 가능한 라이브러리"\*\*와 \*\*"실행 가능한 애플리케이션"\*\*을 명확히 분리하여 유지보수와 확장성을 극대화합니다.

```
rag-project/
├── .env                  # 1. (비밀) 비밀 키, 개인 경로
├── config/               # 2. (설정) 공개 설정값, .env 로더
│
├── src/                  # 3. (라이브러리) 핵심 로직 (재사용 가능)
│   ├── clients/          #    - 외부 API 연동 (Notion, Ollama)
│   ├── common/           #    - 공통 유틸 (텍스트 처리)
│   ├── storage/          #    - 데이터 저장/검색 (Elasticsearch)
│   └── services/         #    - 비즈니스 로직 (RagAgent)
│
├── api/                  # 4. (애플리케이션 1) FastAPI 서버
│   └── main.py
│
├── jobs/                 # 5. (애플리케이션 2) ETL 배치 작업
│   └── batch/
│       └── run_notion_etl.py
│
├── scripts/              # 6. (도구) 모델 워밍업 등 헬퍼 스크립트
│   └── download_models.py
│
└── tests/                # 7. (테스트) 단위/통합 테스트
```

-----

### \#\# 🧩 4. 핵심 구성 요소 (src)

`src/` 라이브러리 패키지는 4개의 주요 클래스로 구성됩니다.

1.  **`src/clients/notion_client.py (NotionClient)`**

      * (구 `notion_loader.py`)
      * Notion API에 연결하여 페이지 블록을 재귀적으로 수집하고 `extract_text_content`로 텍스트를 추출합니다.

2.  **`src/clients/ollama_client.py (OllamaClient)`**

      * Ollama 서버와 통신하는 전용 래퍼 클래스입니다.
      * `get_response` 메서드를 통해 프롬프트를 전송하고 응답을 받습니다.

3.  **`src/common/text_processor.py (TextProcessor)`**

      * (구 `text_processor.py`)
      * `_get_okt` (Konlpy)로 명사를 추출합니다.
      * `_get_kw_model` (KeyBERT)로 키워드를 추출합니다.
      * `chunk_text` 메서드로 "\#\#\# 제목 \#\#\#" 기반 1차 분리 및 `RecursiveCharacterTextSplitter` 2차 청킹을 수행하고, `Document` 객체 리스트를 반환합니다.

4.  **`src/storage/elastic_store.py (ElasticStore)`**

      * (구 `elasticsearch_manager.py` + `search_pipeline.py`)
      * Elasticsearch 클라이언트와 임베딩 모델(Transformers)을 `__init__`에서 초기화합니다.
      * `index_documents`: `_embed_text` (mean pooling)를 호출하여 문서를 ES에 색인합니다.
      * `search_knn`: `knn` 쿼리를 실행하여 `Document` 리스트를 반환합니다.

5.  **`src/services/rag_agent.py (RagAgent)`**

      * (구 `run_hyde_search.py`)
      * RAG의 전체 비즈니스 로직을 담당합니다.
      * `query(question)` 메서드는 다음을 순차적으로 실행합니다:
        1.  `OllamaClient` 호출 (HyDE 가상 답변 생성)
        2.  `ElasticStore.search_knn` 호출 (HyDE 답변으로 컨텍스트 검색)
        3.  `OllamaClient` 재호출 (컨텍스트 + 질문으로 RAG 최종 답변 생성)

-----

### \#\# 🧪 5. 테스트

`pytest`를 사용하여 단위(Unit) 및 통합(Integration) 테스트를 수행합니다.

  * `pytest.ini`에 `pythonpath = .`가 설정되어 있어, `src` 패키지 등을 올바르게 임포트할 수 있습니다.
  * 모든 테스트는 `tests/` 폴더에 위치하며, 실제 API를 호출하는 테스트는 `pytest.mark.integration` 마커가 붙어 있습니다.

**1. 모든 단위 테스트 실행 (빠름, Mock 사용):**

```bash
# (venv)
# 📣 (Intel Mac) C++ 충돌 방지 플래그 포함
KMP_DUPLICATE_LIB_OK=TRUE TQDM_DISABLE=1 pytest -v
```

**2. 통합 테스트만 실행 (느림, 실제 API 호출):**
(Notion, Elasticsearch, Ollama 서버가 모두 실행 중이어야 합니다.)

```bash
# (venv)
KMP_DUPLICATE_LIB_OK=TRUE TQDM_DISABLE=1 pytest -m integration -v
```