# RAGAS E2E 평가 가이드

> **목적**: 전체 RAG 파이프라인(N1~N7)을 실제 운영 경로로 실행하고 RAGAS 4개 메트릭으로 품질을 측정합니다.  
> **위치**: `tests/ragas_e2e/`  
> **표준 judge**: `qwen2.5:72b` (refactoring_log.md 2026-03-21 확정)

---

## 1. 기존 평가(`test_ragas.py`)와의 차이

| 항목 | `test_ragas.py` | `ragas_e2e/test_e2e.py` |
|------|:---:|:---:|
| 검색 | `elasticsearch_store.hybrid_search()` 직접 호출 | `RAGService.process_query()` 경유 |
| HyDE (N1) | ❌ | ✅ |
| Grade Documents (N3) | ❌ | ✅ |
| 웹 검색 fallback (N4→N5) | ❌ | ✅ |
| 생성 프롬프트 | 테스트 전용 하드코딩 | `apps/prompts/chat_prompt.py` (운영 동일) |
| Evidence 식별 (N7) | ❌ | ✅ |
| 파이프라인 진단 | ❌ | ✅ (`ragas_e2e_diagnostics.json`) |

---

## 2. 파이프라인 구조

```
RAGService.process_query()
└── RAGGraph.ainvoke()
    ├── N1: hyde            — HyDE 가상 문서 생성 (rewriter LLM)
    ├── N2: retrieve        — HyDE+query 결합 hybrid_search
    ├── N3: grade_documents — 쿼리-문서 관련성 판정 (grader LLM)
    ├── N4: query_rewrite   — 웹 검색 쿼리 재작성 (web_search=True 시)
    ├── N5: search_web      — Tavily 웹 검색 (web_search=True 시)
    ├── N6: generate        — 답변 생성 (chat_prompt.py, main LLM)
    └── N7: identify_evidence — 하이브리드 근거 식별
```

---

## 3. 패키지 구조

```
tests/ragas_e2e/
├── __init__.py      — 공개 API
├── _helpers.py      — ragas_score, build_ragas_embeddings, save_e2e_result
├── _pipeline.py     — _run_single_sample, build_e2e_dataset
└── test_e2e.py      — TestRAGASE2EGPT, TestRAGASE2EGroq
```

---

## 4. 사전 준비

### 4-1. 환경

```bash
# 평가 전용 가상환경 (langchain-core 버전 충돌 방지)
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

### 4-2. `.env` 필수 항목

```env
ES_HOST=...
ES_API_KEY=...
OLLAMA_HOST=...
CF_ACCESS_CLIENT_ID=...
CF_ACCESS_CLIENT_SECRET=...
TAVILY_API_KEY=...          # N5 웹 검색 필수

# GPT judge 사용 시
OPENAI_API_KEY=...

# Groq judge 사용 시
GROQ_API_KEY=...
GROQ_API_KEY_2=...          # KEY_1 rate limit 분산용 (없으면 KEY_1 fallback)
```

### 4-3. golden_set 파일

| 파일 | 건수 | 용도 |
|------|-----:|------|
| `golden_sets/golden_set_100.json` | 100 | notion 데모 페이지에서 추출 |
| `golden_sets/golden_set_138.json` | 138 | 확장 평가 (1~100: 자동생성, 101~138: open-domain에서 추출) |
| `golden_sets/golden_set_mrc.json` | — | MRC open-domain 특화 |
| `golden_sets/golden_set.json` | — | 레거시 |

> **⚠️ 수작업 데이터 작성 규칙**
>
> `reference_contexts`는 반드시 **`list[str]`** 형식으로 작성해야 합니다.  
> `list[list[str]]`(중첩 배열)로 작성하면 RAGAS `SingleTurnSample` 생성 시 `ValidationError`가 발생하여 전체 테스트가 실패합니다.
>
> ```json
> // ✅ 올바른 형식
> { "reference_contexts": ["컨텍스트 문자열1", "컨텍스트 문자열2"] }
>
> // ❌ 잘못된 형식 (중첩 배열)
> { "reference_contexts": [["컨텍스트 문자열1", "컨텍스트 문자열2"]] }
> ```
>
> 추가 후 검증:
> ```python
> rc = item["reference_contexts"]
> assert all(isinstance(s, str) for s in rc), "reference_contexts 원소가 str이 아님"
> ```

---

## 5. 실행 방법

### 5-1. pytest (CI / 팀 공유)

```bash
# 표준 judge (gpt-4o-mini) — 기본 golden_set
pytest tests/ragas_e2e/test_e2e.py::TestRAGASE2EGPT -v -s

# golden_set 파일 지정
pytest tests/ragas_e2e/test_e2e.py::TestRAGASE2EGPT -v -s \
  --golden-set tests/golden_sets/golden_set_138.json

# Groq judge
pytest tests/ragas_e2e/test_e2e.py::TestRAGASE2EGroq -v -s
```

### 5-2. 단독 실행 (PyCharm 디버깅)

**Run Configuration 설정**

| 항목 | 값 |
|------|-----|
| Script path | `tests/ragas_e2e/test_e2e.py` |
| Parameters | `--judge gpt --golden-set tests/golden_set_100.json` |
| Working directory | 프로젝트 루트 |

```bash
# 터미널 직접 실행
python tests/ragas_e2e/test_e2e.py --judge gpt
python tests/ragas_e2e/test_e2e.py --judge gpt --golden-set tests/golden_sets/golden_set_138.json
```

### 5-3. 디버깅 포인트 (`_pipeline.py` → `_run_single_sample()`)

| 변수 | 확인 내용 | 노드 |
|------|----------|------|
| `all_docs` | 검색된 문서 목록 | N2 |
| `web_search_triggered` | 웹 검색 경로 여부 | N3 |
| `answer` | 운영 프롬프트로 생성된 답변 | N6 |
| `evidence_indices` | 근거 문서 인덱스 | N7 |

---

## 6. 결과 파일

| 파일 | 내용 |
|------|------|
| `tests/ragas_e2e_result.json` | RAGAS 4개 메트릭 점수 + pass/fail |
| `tests/ragas_e2e_diagnostics.json` | 쿼리별 노드 출력 (HyDE, 웹검색 여부, 근거 인덱스, 처리시간) |

### `ragas_e2e_result.json` 예시

```json
{
  "evaluated_at": "2026-04-03T10:00:00",
  "pipeline": "RAGService.process_query() → RAGGraph (N1~N7)",
  "summary": {
    "faithfulness": 0.856,
    "context_precision": 0.948,
    "answer_relevancy": 0.708,
    "context_recall": 0.989,
    "faithfulness_pass": true,
    "context_precision_pass": true,
    "answer_relevancy_pass": true,
    "context_recall_pass": true,
    "total_queries": 100
  }
}
```

---

## 7. 평가 기준 (임계값)

| 메트릭 | 기준 | 측정 원리 |
|--------|-----:|---------|
| Faithfulness | ≥ 70% | 답변 내 주장이 검색 컨텍스트로 뒷받침되는 비율 |
| AnswerRelevancy | ≥ 60% | 역생성 질문 ↔ 원래 질문 임베딩 유사도 |
| ContextPrecision | ≥ 70% | 관련 청크가 검색 결과 상위에 랭크되는지 여부 |
| ContextRecall | ≥ 60% | reference 커버에 필요한 청크 검색 여부 |


---

## 8. 미달 시 개선 방향

| 메트릭 | E2E 관점 원인 | 개선 방향 |
|--------|------------|---------|
| **Faithfulness** | N6 생성 프롬프트가 컨텍스트 외 정보 포함 | `chat_prompt.py` 지시 강화 |
| **AnswerRelevancy** | 답변이 질문과 직접 연결되지 않음 | `chat_prompt.py`에 질문 재인용 지시 추가 (refactoring_log 방향 B) |
| **ContextPrecision** | 관련 없는 문서가 상위에 검색됨 | N1 HyDE 프롬프트 개선 / N2 하이브리드 가중치 조정 |
| **ContextRecall** | 필요한 청크가 검색에서 누락됨 | `TOP_K_RESULTS` 증가 / N3 grade 기준 완화 |

---
