# Project: Advanced Multi-turn RAG System (LangGraph & Elasticsearch)

## 🎯 Project Overview
이 프로젝트는 **LangChain**, **LangGraph**, **Elasticsearch**를 결합한 고성능 멀티턴 RAG 시스템입니다. 로컬 LLM인 **Ollama(Exaone)**를 사용하며, Notion 및 다양한 문서 파일(PDF, DOCX 등)의 자동 인덱싱을 지원합니다.

## 🏗️ Technical Stack & Architecture
- **Framework**: LangChain, LangGraph (State-based workflow)
- **Vector Store**: Elasticsearch (Hybrid Search: Vector 70% + Keyword 30%)
- **LLM & Embedding**: Ollama (Model: `exaone`, Embedding: `nomic-embed-text`)
- **API Layer**: FastAPI (Asynchronous processing)
- **Data Source**: Notion API, Local Files (TXT, PDF, DOCX, MD)

## 📁 Key File Map
- `rag_graph.py`: LangGraph 워크플로우 및 상태 관리 (retrieve -> generate)
- `service.py`: `astream_events` 기반의 스트리밍 비즈니스 로직
- `vector_store.py`: Elasticsearch 통합 및 하이브리드 검색 로직
- `models.py`: Pydantic 기반의 타입 안정성 정의
- `api.py`: FastAPI 엔드포인트 및 요청/응답 처리

## 📝 Coding Standards for Gemini CLI
1. **타입 안정성**: 모든 신규 기능 및 리팩토링 제안 시 `models.py`를 참조하여 Pydantic 모델과 Type Hint를 필수 적용한다.
2. **비동기 우선**: 모든 I/O 작업(DB, API, LLM 호출)은 `async/await`를 사용한다.
3. **스트리밍 대응**: 답변 생성 로직 수정 시 `service.py`의 스트리밍 이벤트 흐름(retrieve_start -> content -> done)을 준수한다.
4. **검색 전략**: 하이브리드 검색의 가중치나 필터링 로직 제안 시 `vector_store.py`의 구현 방식을 최우선으로 고려한다.
5. **프롬프트 관리**: 프롬프트 수정 요청 시 `prompts.py`의 템플릿 구조를 유지하며 페르소나를 적용한다.

## 🤖 Interaction Context
- 대장(User)은 Windows 11 Pro 환경에서 WSL 2(Ubuntu 22.04)를 통해 개발 중이다.
- 답변은 해결책 위주로 간결하게 하며, 코드는 마크다운 블록으로 제공한다.
- 프로젝트의 `standard_template.py` 구조(Logging, Type Hinting 포함)를 기본 코딩 스타일로 삼는다.