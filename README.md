# Multi-turn RAG System

LangGraph 기반 멀티턴 RAG(Retrieval-Augmented Generation) 시스템입니다. Elasticsearch의 하이브리드 검색(Vector + BM25)과 Ollama LLM을 활용하여 한국어 문서에 대한 질의응답을 제공합니다.

## 주요 특징

- **하이브리드 검색**: Vector 유사도 + BM25 키워드 검색 결합
- **멀티턴 대화**: 세션 기반 대화 이력 관리
- **Evidence 추적**: N3 노드의 하이브리드 로직으로 답변 근거 문서 식별
- **비동기 처리**: FastAPI + aiohttp 기반 고성능 비동기 아키텍처

## 아키텍처

### N1-N2-N3 파이프라인

```mermaid
flowchart LR
    subgraph Input
        Q[Query]
    end

    subgraph N1[N1: Retrieve]
        ES[Elasticsearch<br/>Hybrid Search]
    end

    subgraph N2[N2: Generate]
        LLM[Ollama LLM<br/>답변 생성]
    end

    subgraph N3[N3: Identify Evidence]
        EV[LLM + Keyword<br/>하이브리드 검증]
    end

    subgraph Output
        A[Answer + Sources]
    end

    Q --> N1
    N1 -->|retrieved_docs| N2
    N2 -->|answer| N3
    N3 -->|evidence_indices| Output
```

### 시스템 구조

```mermaid
flowchart TB
    subgraph Client
        REQ[HTTP Request]
    end

    subgraph API["FastAPI"]
        ROUTER[Routers]
        SERVICE[RAGService]
    end

    subgraph Graph["LangGraph"]
        N1[retrieve]
        N2[generate]
        N3[identify_evidence]
        N1 --> N2 --> N3
    end

    subgraph External
        ES[(Elasticsearch)]
        OLLAMA[Ollama LLM]
    end

    REQ --> ROUTER --> SERVICE --> Graph
    N1 -.-> ES
    N2 -.-> OLLAMA
    N3 -.-> OLLAMA
```

### 하이브리드 검색 Flow

```mermaid
flowchart LR
    Q[Query] --> ES[ElasticsearchStore]

    ES --> VS[Vector Search<br/>kNN + cosine]
    ES --> KS[Keyword Search<br/>BM25]

    VS --> MERGE[Score Normalization<br/>+ Weighted Merge]
    KS --> MERGE

    MERGE --> TOP[Top-K Documents]
```

**가중치**: `final_score = (vector_score × 0.5) + (keyword_score × 0.5)`

## 설치 방법

### 사전 요구사항

- Python 3.11+
- Elasticsearch 8.x
- Ollama (또는 원격 Ollama 서버)

### macOS

```bash
# 저장소 클론
git clone https://github.com/your-repo/rag-project.git
cd rag-project
git switch notion-analysis
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.sample .env
# .env 파일을 편집하여 설정값 입력

# 서버 실행
cd apps && python main.py
```

### WSL 2 (Windows)

```bash
# WSL 2 Ubuntu 환경에서 실행
sudo apt update && sudo apt install python3.11 python3.11-venv

# 저장소 클론
git clone https://github.com/your-repo/rag-project.git
cd rag-project
git switch notion-analysis
# 가상환경 생성 및 활성화
python3.11 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.sample .env
nano .env  # 설정값 편집

# 서버 실행
cd apps && python main.py
```

## 환경 변수 설정

`.env` 파일에 다음 설정을 추가하세요:

```env
# Elasticsearch
ES_HOST=https://your-elasticsearch-host/
ES_ID=your-api-key-id
ES_API_KEY=your-api-key
ES_INDEX=vector-test-index
VEC_DIMS=1024

# Ollama
OLLAMA_HOST=https://your-ollama-host/
OLLAMA_MODEL=hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:BF16
EMBEDDING_MODEL=bge-m3:latest

# Notion (선택)
NOTION_TOKEN=your-notion-token
NOTION_VERSION=2022-06-28
```

## 사용 방법

### API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/query` | 질의응답 (동기) |
| POST | `/query/stream` | 질의응답 (스트리밍) |
| GET | `/health` | 헬스체크 |
| POST | `/document/add` | 문서 추가 |
| POST | `/search` | 검색 |

### 질의 예시

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "단일성 정체감 장애를 가진 사람의 특징은?",
    "session_id": "test-session",
    "use_history": false
  }'
```

### 응답 예시

```json
{
  "session_id": "test-session",
  "answer": "단일성 정체감 장애 현상은...",
  "sources": [
    {
      "index": 0,
      "content": "문서 내용...",
      "metadata": {"page_id": "..."},
      "is_evidence": true
    }
  ],
  "processing_time": 5.23
}
```

## 테스트

### Golden Set 테스트 결과

| 지표 | 결과 |
|------|------|
| **Hit Rate** | **100%** (10/10) |
| **MRR** | **1.0000** |
| **Evidence 정확도** | **100%** |

### 테스트 실행

```bash
# 단위 테스트
pytest tests/

# Golden Set 평가
python tests/full_evaluation.py
```

## 프로젝트 구조

```
rag-project/
├── apps/
│   ├── api.py              # FastAPI 앱 진입점
│   ├── main.py             # 서버 실행
│   ├── common/
│   │   └── config.py       # 설정 관리
│   ├── graphs/
│   │   └── rag_graph.py    # LangGraph 워크플로우
│   ├── models/
│   │   ├── state.py        # GraphState, Document 등
│   │   ├── request.py      # API 요청 모델
│   │   └── response.py     # API 응답 모델
│   ├── prompts/
│   │   ├── chat_prompt.py
│   │   └── get_evidence_prompt.py
│   ├── routers/
│   │   ├── query.py        # 질의응답 라우터
│   │   ├── document.py     # 문서 관리
│   │   └── system.py       # 시스템 헬스체크
│   ├── services/
│   │   └── service.py      # RAGService
│   ├── stores/
│   │   ├── vector_store.py # Elasticsearch 연동
│   │   └── memory_store.py # 세션 이력 관리
│   └── utils/
│       ├── file_processor.py
│       └── notion_connector.py
├── tests/
│   ├── golden_set.json     # 평가용 질문 세트
│   ├── final_report.md     # 테스트 리포트
│   └── user_test_log.md    # API 테스트 로그
├── .env.sample
├── .gitignore
├── requirements.txt
├── CLAUDE.md               # AI 어시스턴트 지침
└── README.md
```

## 기술 스택

| 분류 | 기술                  |
|------|---------------------|
| API Framework | FastAPI 0.109.0     |
| Orchestration | LangGraph 0.0.20    |
| Search Engine | Elasticsearch 9.1.5 |
| LLM | Ollama (EXAONE 4.0) |
| Embedding | bge-m3 (1024차원)     |
| Data Validation | Pydantic v2         |

## 라이선스

MIT License

## 기여

이슈 및 PR은 언제든 환영합니다.
