# LangChain & LangGraph & Elasticsearch RAG 시스템

## 개요

LangChain, LangGraph, Elasticsearch를 활용한 고성능 멀티턴 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

✅ **타입 안정성**: Pydantic을 활용한 완벽한 타입 힌팅 및 런타임 검증  
✅ **비동기 처리**: async/await를 활용한 고성능 비동기 처리  
✅ **Elasticsearch 통합**: 벡터 검색, 키워드 검색, 하이브리드 검색 지원  
✅ **Notion 통합**: Notion 페이지를 API로 가져와서 자동 인덱싱  
✅ **멀티턴 대화**: 인메모리 저장소를 활용한 세션 기반 대화 이력 관리  
✅ **Ollama 통합**: Exaone 모델을 활용한 로컬 LLM  
✅ **파일 업로드**: TXT, PDF, DOCX, MD 파일 자동 처리 및 인덱싱  
✅ **동시성 제어**: Semaphore를 활용한 요청 동시성 제어  
✅ **설정 관리**: 환경 변수 기반 설정 분리  

## 아키텍처

### 레이어 구조

```
┌─────────────────────────────────────────┐
│          API Layer (api.py)             │
│  - FastAPI 엔드포인트                    │
│  - 요청/응답 처리                        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Service Layer (service.py)         │
│  - 비즈니스 로직                         │
│  - astream_events 기반 스트리밍         │
│  - 메모리 관리                           │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Graph Layer (rag_graph.py)         │
│  - LangGraph 워크플로우                 │
│  - 노드: retrieve, generate             │
│  - 상태 관리                             │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Infrastructure Layer                 │
│  - Elasticsearch (vector_store.py)      │
│  - Memory Store (memory_store.py)       │
│  - Prompts (prompts.py)                 │
└─────────────────────────────────────────┘
```

### 스트리밍 이벤트 흐름

`service.py`에서 `astream_events`를 사용하여 다음 이벤트를 생성합니다:

1. **retrieve_start**: 문서 검색 시작
2. **retrieve_end**: 문서 검색 완료 (찾은 문서 수 포함)
3. **generate_start**: 답변 생성 시작
4. **content**: 답변 내용 (토큰 단위 스트리밍)
5. **generate_end**: 답변 생성 완료
6. **done**: 전체 처리 완료

## 프로젝트 구조

```
rag-system/
├── config.py                    # 설정 관리
├── models.py                    # Pydantic 데이터 모델
├── prompts.py                   # 프롬프트 템플릿 관리
├── memory_store.py              # 인메모리 대화 저장소
├── vector_store.py              # Elasticsearch 벡터 스토어
├── rag_graph.py                 # LangGraph RAG 워크플로우
├── service.py                   # RAG 서비스 레이어
├── file_processor.py            # 파일 처리 유틸리티
├── notion_connector.py          # Notion API 연동
├── api.py                       # FastAPI 서버
├── example_client.py            # 사용 예시 클라이언트
├── streaming_client_example.py  # 스트리밍 클라이언트 예시
├── custom_prompts_example.py    # 프롬프트 커스터마이징 예시
├── requirements.txt             # 의존성
├── .env.example                 # 환경 변수 예시
└── README.md                    # 문서
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Elasticsearch 설치 및 실행

```bash
# Docker를 사용하는 경우
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.1

# 또는 직접 설치
# https://www.elastic.co/downloads/elasticsearch
```

### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Exaone 모델 다운로드
ollama pull exaone

# 임베딩 모델 다운로드
ollama pull nomic-embed-text
```

### 4. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일을 필요에 맞게 수정
```

**중요: Notion 통합 설정**

Notion API를 사용하려면 Integration을 생성하고 토큰을 발급받아야 합니다:

1. [Notion Integrations](https://www.notion.so/my-integrations) 페이지 접속
2. "New integration" 클릭
3. Integration 이름 설정 (예: "RAG System")
4. "Submit" 클릭하여 Integration 생성
5. "Internal Integration Token" 복사
6. `.env` 파일의 `NOTION_TOKEN`에 토큰 붙여넣기
7. Notion에서 가져올 페이지를 열고 우측 상단 `...` → "Add connections" → 생성한 Integration 선택

### 5. 서버 실행

```bash
python api.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

## API 사용 예시

### 1. 텍스트 문서 추가

```bash
curl -X POST "http://localhost:8000/documents/add" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Elasticsearch는 확장 가능한 오픈소스 검색 및 분석 엔진입니다.",
    "metadata": {"source": "docs", "topic": "elasticsearch"}
  }'
```

### 2. 파일 업로드

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf" \
  -F 'metadata={"category": "technical", "author": "John Doe"}'
```

### 3. 질의응답 (멀티턴 대화)

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "query": "Elasticsearch의 특징은 무엇인가요?",
    "use_history": true
  }'
```

### 4. 하이브리드 검색

```bash
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "검색 엔진",
    "top_k": 5,
    "filters": {"topic": "elasticsearch"}
  }'
```

### 5. 벡터 검색

```bash
curl -X POST "http://localhost:8000/search/vector" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "분산 검색 시스템",
    "top_k": 3
  }'
```

### 6. 키워드 검색

```bash
curl -X POST "http://localhost:8000/search/keyword" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "오픈소스",
    "top_k": 5
  }'
```

### 7. 문서 수 조회

```bash
curl -X GET "http://localhost:8000/documents/count"
```

### 8. 대화 이력 조회

```bash
curl -X GET "http://localhost:8000/sessions/user123/history"
```

## Python 클라이언트 예시

```python
import asyncio
import aiohttp
import aiofiles

async def main():
    async with aiohttp.ClientSession() as session:
        # Notion 페이지 가져오기
        async with session.post(
            "http://localhost:8000/notion/import",
            json={
                "page_id": "123abc456def",
                "metadata": {"category": "research"},
                "recursive": True
            }
        ) as resp:
            result = await resp.json()
            print("Notion 가져오기:", result)
        
        # 파일 업로드
        data = aiohttp.FormData()
        async with aiofiles.open('document.pdf', 'rb') as f:
            content = await f.read()
            data.add_field('file', 
                          content,
                          filename='document.pdf',
                          content_type='application/pdf')
            data.add_field('metadata', '{"category": "research"}')
        
        async with session.post(
            "http://localhost:8000/documents/upload",
            data=data
        ) as resp:
            result = await resp.json()
            print("파일 업로드:", result)
        
        # 하이브리드 검색
        async with session.post(
            "http://localhost:8000/search/hybrid",
            json={
                "query": "인공지능 기술",
                "top_k": 3
            }
        ) as resp:
            result = await resp.json()
            print(f"검색 결과: {result['total_hits']}개")
            for r in result['results']:
                print(f"- {r['content'][:100]}... (score: {r['score']})")
        
        # 질의응답
        async with session.post(
            "http://localhost:8000/query",
            json={
                "query": "검색된 내용을 요약해주세요",
                "use_history": True
            }
        ) as resp:
            result = await resp.json()
            print("답변:", result["answer"])
            print("세션 ID:", result["session_id"])  # 다음 요청에 사용
            
            # 같은 세션으로 후속 질문
            session_id = result["session_id"]
        
        async with session.post(
            "http://localhost:8000/query",
            json={
                "session_id": session_id,
                "query": "좀 더 자세히 설명해줘",
                "use_history": True
            }
        ) as resp:
            result = await resp.json()
            print("후속 답변:", result["answer"])

asyncio.run(main())
```

## 검색 방식 비교

### 1. 벡터 검색 (Vector Search)
- 의미적 유사도 기반 검색
- 임베딩 벡터의 코사인 유사도 계산
- 동의어나 유사 표현도 잘 검색됨

### 2. 키워드 검색 (Keyword Search)
- 전통적인 전문 검색 (Full-text search)
- 정확한 키워드 매칭
- 빠른 검색 속도

### 3. 하이브리드 검색 (Hybrid Search)
- 벡터 검색 + 키워드 검색 결합
- 각 검색 결과를 가중 평균하여 최종 점수 계산
- 기본 가중치: 벡터 70%, 키워드 30%
- 가장 균형잡힌 검색 결과

## API 문서

서버 실행 후 다음 URL에서 자동 생성된 API 문서를 확인할 수 있습니다:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 프롬프트 커스터마이징

시스템 프롬프트를 쉽게 수정할 수 있도록 `prompts.py` 파일에서 관리됩니다.

### 프롬프트 템플릿 종류

**prompts.py**에 다음 프롬프트들이 정의되어 있습니다:

1. **SYSTEM_PROMPT_WITH_HISTORY**: 대화 이력이 있을 때 사용
2. **SYSTEM_PROMPT_WITHOUT_HISTORY**: 대화 이력이 없을 때 사용
3. **SUMMARIZATION_PROMPT**: 문서 요약용
4. **KEYWORD_EXTRACTION_PROMPT**: 키워드 추출용
5. **QUERY_ENHANCEMENT_PROMPT**: 검색 쿼리 개선용

### 기본 프롬프트 수정

```python
# prompts.py 파일에서 직접 수정
class RAGPrompts:
    SYSTEM_PROMPT_WITH_HISTORY = """당신의 커스텀 프롬프트...
    
컨텍스트:
{context}

답변 시 다음을 지켜주세요:
1. 첫 번째 규칙
2. 두 번째 규칙
...
"""
```

### 동적 프롬프트 생성

```python
from prompts import rag_prompts

# 커스텀 시스템 메시지
custom_prompt = rag_prompts.get_custom_prompt(
    system_message="당신은 전문 기술 컨설턴트입니다...",
    include_history=True
)
```

### 도메인별 특화 프롬프트

`custom_prompts_example.py` 파일에서 다양한 도메인별 프롬프트 예시를 확인할 수 있습니다:

- **기술 문서 전문가**: 코드 예시 포함, 상세한 기술 설명
- **비즈니스 컨설턴트**: ROI 관점, 실행 가능한 액션 아이템
- **교육 튜터**: 쉬운 설명, 단계별 학습
- **의료 정보 제공자**: 정확한 의료 정보, 전문의 상담 권장
- **법률 정보 제공자**: 중립적 법률 정보, 법률가 상담 권장
- **창의적 작가**: 스토리텔링, 독창적 표현

### RAG 그래프에서 프롬프트 변경

`rag_graph.py`에서 사용할 프롬프트를 변경하려면:

```python
# rag_graph.py의 _generate_node 메서드에서
from prompts import rag_prompts

# 기본 프롬프트 사용
prompt = rag_prompts.get_qa_prompt_with_history()

# 또는 커스텀 프롬프트 사용
prompt = rag_prompts.get_custom_prompt(
    system_message="당신만의 시스템 메시지...",
    include_history=True
)
```

## 성능 최적화

### Elasticsearch 설정

```python
# .env 파일
EMBEDDING_DIM=768  # 임베딩 차원 (모델에 따라 다름)
```

### 청킹 설정

```python
# .env 파일
CHUNK_SIZE=1000      # 청크 크기
CHUNK_OVERLAP=200    # 청크 오버랩
```

### 검색 결과 수

```python
# .env 파일
TOP_K_RESULTS=3      # 기본 검색 결과 수
```

### 동시성 제어

```python
# .env 파일
MAX_CONCURRENT_REQUESTS=5
```

## 지원하는 문서 소스

### 파일 업로드
- **TXT**: 일반 텍스트 파일
- **PDF**: PDF 문서 (pypdf 사용)
- **DOCX**: Microsoft Word 문서 (python-docx 사용)
- **MD**: Markdown 파일

### Notion 통합
- **페이지 내용**: 제목, 단락, 헤딩, 리스트 등
- **블록 타입**: paragraph, heading, bulleted_list, numbered_list, to_do, code, quote, callout 등
- **재귀적 가져오기**: 하위 블록과 중첩된 내용 자동 처리
- **메타데이터**: 페이지 제목, URL, 생성/수정 시간 자동 저장

## 기술 스택

- **FastAPI**: 비동기 웹 프레임워크
- **LangChain**: LLM 애플리케이션 프레임워크
- **LangGraph**: 상태 기반 워크플로우 엔진
- **Elasticsearch**: 분산 검색 및 분석 엔진
- **Notion API**: 노트 콘텐츠 통합
- **Ollama**: 로컬 LLM 실행 환경
- **Pydantic**: 데이터 검증 및 타입 힌팅
- **aiohttp**: 비동기 HTTP 클라이언트

## Elasticsearch vs ChromaDB

### Elasticsearch 장점
✅ 프로덕션 레벨 확장성  
✅ 다양한 검색 방식 (벡터, 키워드, 하이브리드)  
✅ 강력한 필터링 및 집계 기능  
✅ 분산 아키텍처 지원  
✅ 실시간 인덱싱  
✅ 풍부한 모니터링 및 관리 도구  

### ChromaDB 장점
✅ 간단한 설치 및 설정  
✅ 가벼운 임베딩 전용 DB  
✅ 로컬 개발에 적합  

## 주의사항

1. **Elasticsearch 실행**: API 실행 전 Elasticsearch 서비스가 실행 중이어야 합니다
2. **Ollama 실행**: Ollama 서비스가 실행 중이어야 합니다
3. **모델 다운로드**: Exaone 및 nomic-embed-text 모델이 다운로드되어 있어야 합니다
4. **Notion 토큰**: Notion 통합을 사용하려면 `.env`에 `NOTION_TOKEN` 설정 필요
5. **Notion 권한**: 가져올 페이지에 Integration 연결 필요
6. **메모리 관리**: 인메모리 대화 저장소는 서버 재시작 시 초기화됩니다
7. **파일 업로드**: 대용량 파일 처리 시 타임아웃 설정 필요
8. **임베딩 차원**: 사용하는 임베딩 모델의 차원에 맞게 `EMBEDDING_DIM` 설정 필요

## 트러블슈팅

### Elasticsearch 연결 오류
```bash
# Elasticsearch 상태 확인
curl http://localhost:9200

# Docker 컨테이너 로그 확인
docker logs elasticsearch
```

### Ollama 연결 오류
```bash
# Ollama 서비스 상태 확인
ollama list

# Ollama 재시작
ollama serve
```

### Notion API 오류
```bash
# 1. 토큰이 올바르게 설정되었는지 확인
echo $NOTION_TOKEN

# 2. Integration이 페이지에 연결되었는지 확인
# Notion 페이지 → 우측 상단 ... → Connections 확인

# 3. 페이지 ID 확인
# URL: https://www.notion.so/My-Page-123abc...
# 페이지 ID: 123abc... (32자리 영숫자)
```

## 라이선스

MIT License
