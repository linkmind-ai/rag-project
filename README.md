### 개요
이 레포는 Notion 문서를 수집·정제하고, 한국어 키워드/청크 생성 → 벡터 임베딩 → Elasticsearch(k‑NN) 색인 → HyDE 확장 질의 → ES 검색 → Ollama로 최종 응답 생성까지의 RAG 파이프라인을 담고 있습니다.

---

### 최상위 디렉터리 구조와 역할
- `config/`
  - `.env`에서 비밀값/경로를 로드하고 전역 설정(상수, 모델명 등)을 노출합니다.
  - 파일: `config/__init__.py`
- `notion_loader.py`
  - Notion API로 페이지/블록을 재귀 수집하고, 텍스트를 추출합니다.
- `text_processor.py`
  - 한국어 전처리: 제목 헤더 분할, `Okt` 명사 추출, `KeyBERT` 키워드 추출, `RecursiveCharacterTextSplitter`로 청킹.
- `elasticsearch_manager.py`
  - ES 클라이언트 생성, 임베딩 모델 로드(HF Transformers), 텍스트 임베딩, 인덱스 생성/삭제, 색인/조회 유틸.
- `search_pipeline.py`
  - ES 9.x k‑NN 검색(벡터 필드)을 호출해 `langchain.schema.Document` 목록으로 반환.
- `run_hyde_search.py`
  - HyDE(가상 응답)로 질의를 확장 → 임베딩 → ES k‑NN 검색 → Ollama로 최종 답변 생성.
- `main_workflow.py`
  - (오프라인 준비 단계) Notion → 파일 저장 → 텍스트 로드/청킹/키워드 → ES 인덱싱까지의 배치 파이프라인.
- `tests/`
  - 단위/통합 테스트. 실제 외부 자원(ES/Notion/Ollama)을 쓰는 테스트는 `integration` 마커.
- 그 외
  - `utils/`, `rag_pipe/`, `logs/`: 보조 유틸/파이프라인/로그 디렉터리(현재 세부 구현은 제공 파일에 없음).
  - `requirements.txt`, `pytest.ini`, `README.md` 등 프로젝트 설정/문서.

---

### 설정과 환경변수 (`config/__init__.py`)
- .env에서 로드되는 주요 변수
  - `NOTION_TOKEN`, `ES_API_KEY`, `ES_ID`, `ES_HOST`, `OLLAMA_HOST`
  - 경로: `JSON_SAVE_PATH`, `TXT_SAVE_PATH`, `CHROMA_DB_PATH`
- 상수
  - `NOTION_VERSION`, `ES_INDEX_NAME = "vector-test-index"`, `ES_EMBEDDING_DIMS = 384`
  - 모델: `KEYBERT_MODEL`, `ES_EMBEDDING_MODEL`, `CHROMA_EMBEDDING_MODEL`, `RERANKER_MODEL`, `OLLAMA_MODEL`
- 로드 실패 시 바로 예외를 던지도록 방어적으로 설계되어 있습니다.

---

### 데이터 파이프라인 1: 오프라인 색인(배치)
진입점: `main_workflow.py`
1) Notion 수집 및 저장
- `run_notion_to_file()`
  - 입력된 Notion `page_id`로 `notion_loader.get_page_content`/`fetch_all_blocks` 수행
  - 페이지 메타(JSON)와 추출 텍스트(TXT)를 각각 `JSON_SAVE_PATH`/`TXT_SAVE_PATH`에 저장

2) 텍스트 처리 → 청킹 → ES 색인
- `run_process_and_index()`
  - `text_processor.load_text_from_file(TXT_SAVE_PATH)`
  - `text_processor.chunk_text_with_recursive_splitter(...)`로 문서 헤더 기반 세분화 + 키워드 메타 부착
  - `elasticsearch_manager.get_es_client()` / `get_embedding_model()`로 ES와 임베딩 모델 준비
  - `elasticsearch_manager.create_es_index(...)`로 인덱스 생성 후 `index_documents(...)`로 색인
  - 색인 완료 후 `search_all_docs(...)`로 등록 내용 확인

임베딩/ES 세부
- 임베딩: `transformers`의 `AutoTokenizer`/`AutoModel`로 CLS 평균(pooling)하여 리스트 반환
- 인덱스 매핑: `dense_vector(dims=ES_EMBEDDING_DIMS, similarity=cosine, index=True)`

---

### 데이터 파이프라인 2: 온라인 검색 + 생성(RAG)
진입점: `run_hyde_search.py`
1) HyDE(확장 질의)
- Ollama(`config.OLLAMA_HOST`, `config.OLLAMA_MODEL`)로 질문에 대한 “요약된 가상 답변” 생성
- 실패 시 원문 질문으로 대체

2) 벡터화 후 ES k‑NN 검색
- `elasticsearch_manager.embed_text(...)`로 확장 질의 임베딩
- `search_pipeline.search_es_knn(es_client, query_vector, index_name=ES_INDEX_NAME, k=3)` 호출
- 상위 k개를 `Document` 리스트로 수집

3) 최종 답변 생성
- 컨텍스트를 포함한 프롬프트로 Ollama `client.chat()` 호출 → 문자열 응답 반환

---

### 핵심 모듈별 요약
- `notion_loader.py`
  - `get_page_content`, `fetch_all_blocks`, `extract_text_content`
  - Notion 블록 트리를 재귀 순회, 다양한 블록 타입을 텍스트에 반영(제목, 리스트, 코드 등)
- `text_processor.py`
  - `Okt`로 명사 추출 → `KeyBERT(SBERT)`로 키워드 추출
  - `split_text_with_headers`로 "### 제목 ###" 기반 분할 + 키워드 메타 생성
  - `chunk_text_with_recursive_splitter`로 세부 청킹 후 `langchain.Document`로 반환
- `elasticsearch_manager.py`
  - `get_es_client`(API Key, TLS 검증 off 옵션), `get_embedding_model`
  - `embed_text`(mean pooling), `create_es_index`, `index_documents`, `search_all_docs`, `delete_all_docs`
- `search_pipeline.py`
  - ES 9.x `knn` 쿼리 작성/호출, `_source_excludes=['embedding']`로 네트워크 비용 절감
- `run_hyde_search.py`
  - HyDE → 임베딩 → ES 검색 → Ollama 응답까지 end‑to‑end

---

### 테스트 구조(`tests/`)
- `test_search_pipeline.py`
  - 단위 테스트: ES 클라이언트 mock으로 `knn` 파라미터/반환 형식(`Document`) 검증
- `test_integration.py` (pytest `-m integration`)
  - Notion API 연결/수신 확인
  - ES API 연결 및 버전 확인(예: `9.1.5`)
  - ES k‑NN 색인→검색→삭제 사이클
  - 전체 RAG 파이프라인: `run_hyde_search.run_hyde_pipeline(...)`가 문자열 응답을 반환하는지 검증
- `pytest.ini`
  - `integration` 마커 정의, `pythonpath=.`
  - 경고 필터(예: `threadpoolctl` 런타임 경고, `urllib3` 보안 경고 무시)

---

### 실행 순서(예시)
1) 환경 준비
- `.env`에 필수 키/경로 설정: `NOTION_TOKEN`, `ES_HOST`, `ES_ID`, `ES_API_KEY`, `OLLAMA_HOST`, `TXT_SAVE_PATH`, `JSON_SAVE_PATH` 등
- ES 9.x 인스턴스와 Ollama 서버 가동

2) 데이터 색인(오프라인)
- `python -m main_workflow`에서 `run_notion_to_file()` 실행 후 `run_process_and_index()`로 ES에 색인
  - 또는 스크립트 내부에서 순차 호출 로직을 구성

3) 검색/응답(온라인)
- `python -m run_hyde_search` 스타일로 진입하거나, 코드에서 `run_hyde_search.run_hyde_pipeline(question, es_client, tokenizer, model)` 호출

4) 테스트
- 단위: `pytest tests/test_search_pipeline.py -q`
- 통합: `pytest -m integration -q` (실제 Notion/ES/Ollama 필요, 색인 데이터 선행 필요)

---

### 유의사항(잠재 이슈)
- 함수 시그니처 불일치 가능성
  - `elasticsearch_manager.create_es_index(es_client, index_name, vec_dims)` 등은 인자 필요하지만, `main_workflow.py`는 현재 인자 없이 호출하는 부분이 있어 보입니다. 실제 실행 전 해당 호출부를 최신 시그니처에 맞게 수정이 필요합니다.
- Ollama 프롬프트 컨텍스트
  - `run_hyde_search.py`에서 `contexts_docs`(`Document` 리스트)를 문자열로 바로 포맷팅합니다. 모델에 전달될 최종 문자열 형식(예: join/요약)이 필요할 수 있습니다.
- 환경 변수 강제 검증
  - `config`가 import 시점에 부족한 값이 있으면 즉시 예외를 던집니다. 로컬에서 빠르게 일부 기능만 테스트하려면 mock/우회가 필요할 수 있습니다.

---

### 한눈에 보는 데이터 흐름
- 색인 파이프라인: Notion → `notion_loader` → 텍스트(TXT) → `text_processor`(키워드/청킹) → `embed_text` → `index_documents` → ES
- 검색 파이프라인: 질문 → HyDE(Ollama) → `embed_text` → ES k‑NN → 컨텍스트 → Ollama 최종 답변

필요하시면 실제 실행 명령, .env 샘플, 또는 시그니처 정합성 수정 포인트를 함께 정리해 드리겠습니다.