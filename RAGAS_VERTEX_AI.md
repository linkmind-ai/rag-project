# Ragas + Vertex AI 실행 가이드

이 프로젝트는 기존 LangGraph RAG 서버를 실행한 뒤, 별도 평가 스크립트로 `ragas`를 돌리는 방식입니다.

구성은 이렇게 나뉩니다.

- RAG 앱 실행: LangGraph + Elasticsearch + Ollama
- Ragas 평가: Vertex AI API를 사용하는 Gemini judge

## 1. `.env` 설정

`.env`에 아래 값을 넣어주세요.

```env
VERTEX_API_KEY=YOUR_NEW_VERTEX_API_KEY
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=true
RAGAS_VERTEX_MODEL=gemini-2.5-flash
RAGAS_VERTEX_EMBEDDING_MODEL=gemini-embedding-001
RAGAS_QUERY_API_URL=http://127.0.0.1:8000/query
RAGAS_GOLDEN_SET=tests/golden_set_100.json
RAGAS_OUTPUT_PREFIX=.benchmarks/ragas_vertex_eval_golden_set_100
RAGAS_CACHE_PATH=.benchmarks/ragas_vertex_eval_golden_set_100_samples.json
```

## 2. 서버 환경 설치

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ragas-vertex.txt
```

## 3. RAG 서버 실행

터미널 1:

```powershell
.venv\Scripts\activate
cd apps
python main.py
```

기본 평가 대상 API:

- `http://127.0.0.1:8000/query`

## 4. 첫 평가 실행

터미널 2:

```powershell
.venv\Scripts\activate
python scripts\ragas_vertex_eval.py --golden-set tests\golden_set_100.json --limit 100
```

이 첫 실행에서는:

1. `golden_set_100.json`의 질문을 `/query`에 보냅니다.
2. 각 질문의 답변과 retrieved contexts를 캐시에 저장합니다.
3. 그 캐시를 바탕으로 `ragas` 평가를 실행합니다.

## 5. 캐시와 결과 파일

기본 생성 파일:

- CSV 결과: `.benchmarks/ragas_vertex_eval_golden_set_100.csv`
- JSON 요약: `.benchmarks/ragas_vertex_eval_golden_set_100_summary.json`
- 수집 캐시: `.benchmarks/ragas_vertex_eval_golden_set_100_samples.json`

캐시 파일에는 `/query` 호출 결과가 저장됩니다.
즉, 이후에는 같은 질문들을 다시 수집하지 않고 `ragas`만 다시 돌릴 수 있습니다.

## 6. `--resume`으로 재평가

한 번 캐시를 만들고 나면, 다음부터는 `/query`를 다시 치지 않고 평가만 다시 할 수 있습니다.

```powershell
.venv\Scripts\activate
python scripts\ragas_vertex_eval.py --resume --cache-path .benchmarks/ragas_vertex_eval_golden_set_100_samples.json
```

## 7. 메트릭 선택

사용 가능한 메트릭:

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_correctness`
- `answer_relevancy`

`answer_correctness`가 느리거나 timeout이 날 때는 이렇게 제외하고 먼저 돌릴 수 있습니다.

```powershell
python scripts\ragas_vertex_eval.py --resume --cache-path .benchmarks/ragas_vertex_eval_golden_set_100_samples.json --metrics context_precision,context_recall,faithfulness,answer_relevancy
```

## 8. 병렬도와 재시도 조절

429 `RESOURCE_EXHAUSTED`가 자주 나면 병렬도를 낮춰서 실행하세요.

```powershell
python scripts\ragas_vertex_eval.py --resume --cache-path .benchmarks/ragas_vertex_eval_golden_set_100_samples.json --metrics context_precision,context_recall,faithfulness,answer_relevancy --limit 10 --max-workers 1 --batch-size 1 --ragas-timeout 300 --ragas-max-retries 15 --ragas-max-wait 120
```

## 9. Persona 넣고 평가하기

이제 페르소나 프로필을 서버 세션에 미리 심은 뒤 평가할 수 있습니다.
예시 파일은 [persona_eval.sample.json](C:/Users/yss63/rag-project/persona_eval.sample.json) 입니다.

지원되는 persona 키:

- `preferred_topics`
- `avoid_topics`
- `response_style`
- `factuality_bias`
- `explicit_notes`

예시 실행:

```powershell
python scripts\ragas_vertex_eval.py --golden-set tests\golden_set_100.json --limit 10 --persona-file persona_eval.sample.json --output-prefix .benchmarks/ragas_vertex_eval_persona --cache-path .benchmarks/ragas_vertex_eval_persona_samples.json
```

중요:

- persona를 바꿔서 평가할 때는 새 `--output-prefix`와 새 `--cache-path`를 쓰는 것이 안전합니다.
- `--resume`은 같은 persona로 만든 캐시에만 사용해야 합니다.

## 10. 현재 persona 반영 방식

현재 추가한 persona 평가는 `세션 프로필`을 미리 주입하는 방식입니다.
즉 다음 항목이 retrieval/generation에 반영됩니다.

- `preferred_topics`
- `response_style`
- `explicit_notes`

샘플 간 history 오염은 막기 위해, 기본적으로 각 평가 샘플은 별도 세션으로 수집합니다.
