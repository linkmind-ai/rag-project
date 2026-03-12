# RAG 파이프라인 품질 평가 보고서

> 작성일: 2026-03-12
> 브랜치: `notion-analysis`
> 데이터 출처: Notion 연동 문서 (도서 독후감/요약 10종)

---

## 1. 왜 RAG를 평가하는가?

이 프로젝트는 Notion에 저장된 문서를 기반으로 LLM이 질문에 답변하는 RAG 파이프라인을 사용합니다.
평가는 두 단계로 나뉩니다.

```
[Phase 1] 검색 품질 평가               [Phase 2] 생성 품질 평가
──────────────────────────              ──────────────────────────
질문 (Golden Set)                       질문
    │                                       │
    ▼                                       ▼
Elasticsearch 하이브리드 검색           ES 검색 → contexts 확보
    │                                       │
    ▼                                       ▼
Top-K 청크 반환                         Ollama EXAONE → 답변 생성
    │                                       │
    ▼                                       ▼
정답 page_id 포함 여부 확인             Groq (judge) → RAGAS 자동 채점
    │                                       │
    ├─ Hit Rate : 적중률                    ├─ Faithfulness : 원문 충실도
    └─ MRR      : 정답 순위                 └─ Answer Relevancy : 답변 관련성
```

> **역할 분리 원칙**
> - 답변 생성: **Ollama EXAONE-4.0-1.2B** — 실제 운영 시스템과 동일한 모델로 측정
> - RAGAS 평가(judge): **Groq llama-3.3-70b-versatile** — 답변이 컨텍스트에 근거하는지, 질문에 맞는지 판단

---

## 2. 평가 지표 설명

### 2-1. Hit Rate @K — "Top-K 안에 정답 문서가 있는가?"

| 개념 | 설명 |
|------|------|
| 의미 | 10개 질문 중 몇 개에서 정답 문서가 Top-K 결과에 포함되었는가 |
| 기준 | **@3 ≥ 60%, @5 ≥ 70%** |

### 2-2. MRR @K — "정답이 몇 번째에 나오는가?"

| 개념 | 설명 |
|------|------|
| 의미 | 정답 순위의 역수 평균 (1위=1.0, 2위=0.5, 3위=0.33) |
| 기준 | **@5 ≥ 0.40** |

### 2-3. Faithfulness — "답변이 원문에 근거하는가?"

| 개념 | 설명 |
|------|------|
| 의미 | 답변의 각 주장(claim)이 검색된 청크로 뒷받침 가능한가 |
| 계산 | 뒷받침 가능한 주장 수 ÷ 전체 주장 수 |
| 기준 | **≥ 70%** |
| 위험 | 낮으면 LLM이 컨텍스트 무시하고 훈련 기억값(hallucination)을 출력 |

### 2-4. Answer Relevancy — "답변이 질문에 맞는가?"

| 개념 | 설명 |
|------|------|
| 의미 | 답변 내용이 질문을 제대로 다루고 있는가 |
| 계산 | 답변으로 역생성한 질문과 원래 질문의 코사인 유사도 (bge-m3 임베딩) |
| 기준 | **≥ 70%** |

---

## 3. Golden Set — 시험 문제 설계

총 **10개 질문** (Notion 연동 도서 독후감/요약 문서 기반)

| # | 질문 | 핵심 키워드 |
|---|------|------------|
| 1 | 단일성 정체감 장애 특징과 저자 조언 | 단일성 정체감 장애, 공산주의자가 온다 |
| 2 | 좋은 관찰과 나쁜 관찰의 차이점 | 좋은 관찰, 나쁜 관찰, 가설 갱신 |
| 3 | 인지 편향이 관찰을 방해하는 이유 | 인지 편향, 확증편향 |
| 4 | 카밀라 팡의 두려움/완벽주의 비유 | 빛의 굴절, 열역학 |
| 5 | 소설 목화의 꿈속 능력 | 단 한사람, 목화, 꿈속 죽음 |
| 6 | 창업 프로토타입 원칙 | 프로토타입, 직접 하라 |
| 7 | 이커머스 데이터 마술봉 비유 | 대형 플랫폼, 데이터, PB |
| 8 | ANT 이론 약점과 대안 | ANT, 하먼, 객체중심존재론 |
| 9 | 연령인식의 노화 영향 | 연령인식, ageism |
| 10 | 추상미술과 뇌의 정보 처리 | 상향식, 하향식 처리 |

---

## 4. Phase 1 결과 — 검색 품질 (Hit Rate / MRR)

### 결과

| 지표 | 결과 | 기준 | 판정 |
|------|-----:|-----:|:----:|
| Hit Rate @3 | **100%** | ≥ 60% | ✅ 합격 |
| Hit Rate @5 | **100%** | ≥ 70% | ✅ 합격 |
| MRR @5 | **1.000** | ≥ 0.40 | ✅ 합격 |

> **MRR 1.000 의미**: 10개 질문 모두에서 정답 문서가 1순위로 검색됨.
> Elasticsearch 하이브리드 검색(벡터 kNN + BM25, 각 0.5 가중치)이 이 데이터셋에서 완벽하게 동작.

---

## 5. Phase 2 결과 — 생성 품질 (Faithfulness / Answer Relevancy)

### 평가 방법

```
1. ES hybrid_search로 각 질문별 Top-5 청크 검색
2. Ollama EXAONE-4.0-1.2B가 해당 청크만 보고 한국어 답변 생성
3. Groq(llama-3.3-70b)가 RAGAS judge로 두 지표 자동 채점
   - Faithfulness : 주장별 컨텍스트 뒷받침 가능 여부 판정
   - Answer Relevancy : bge-m3 임베딩으로 질문-답변 유사도 측정
```

### 결과 (2026-03-12 측정)

| 지표 | 결과 | 기준 | 판정 |
|------|-----:|-----:|:----:|
| Faithfulness | **61.6%** | ≥ 70% | ❌ 미달 |
| Answer Relevancy | **78.7%** | ≥ 70% | ✅ 합격 |

### 샘플별 답변 품질 관찰

답변들을 분석해보면 다음 패턴이 나타납니다:

**Faithfulness 미달 패턴 (컨텍스트 이탈 사례)**

| 질문 | 문제 패턴 |
|------|----------|
| Q1 단일성 정체감 장애 | "타인 관찰 기술 향상"이라는 컨텍스트에 없는 조언 추가 |
| Q4 목화 꿈속 능력 | "선택의 기로"라는 컨텍스트에 없는 표현 사용 |
| Q8 ANT 이론 | "공생적 관계"라는 컨텍스트에 없는 개념 추가 |
| Q9 연령인식 | "운동이나 인지 활동 유지"라는 구체적 예시를 훈련 기억에서 생성 |

**공통 원인**: EXAONE-4.0-1.2B 소형 모델이 컨텍스트를 보완하려는 경향 → 훈련 기억값을 자연스럽게 삽입

---

## 6. 기술 스택

| 컴포넌트 | 기술 | 역할 |
|---------|------|------|
| 문서 저장 | Elasticsearch 9.1.5 (kNN + BM25) | Notion 청크 저장 및 하이브리드 검색 |
| 임베딩 | bge-m3 (Ollama, 1024차원) | 텍스트 → 벡터 변환 |
| 답변 생성 LLM | EXAONE-4.0-1.2B (Ollama) | 컨텍스트 기반 한국어 답변 생성 |
| RAGAS judge LLM | Groq / llama-3.3-70b-versatile | Faithfulness / Answer Relevancy 평가 |
| RAGAS 임베딩 | bge-m3 (Ollama) | Answer Relevancy 코사인 유사도 측정 |
| 네트워크 보안 | Cloudflare Access (Service Token) | ollama.nabee.ai.kr 인증 |

---

## 7. 개선 방향

### 7-1. [단기 / 높음] 프롬프트 강화로 Faithfulness 개선

**현재 프롬프트 (`apps/prompts/chat_prompt.py`)**:
```
컨텍스트에 있는 정보를 우선적으로 활용하세요.
확실하지 않은 내용은 모른다고 답변하세요.
```

**문제**: "우선적으로"라는 표현이 컨텍스트 외 정보 사용을 암묵적으로 허용함.

**개선 방향**:
```
컨텍스트에 포함된 내용만을 사용하여 답변하세요.
컨텍스트에 없는 내용은 절대 추가하지 마세요.
모르는 경우 "제공된 자료에서 찾을 수 없습니다"라고 답하세요.
```

**기대 효과**: Faithfulness 61.6% → 75%+ 목표
**작업 파일**: `apps/prompts/chat_prompt.py`

---

### 7-2. [단기 / 중간] 청크 크기 및 오버랩 최적화

**현재 설정** (`apps/common/config.py`):
```
CHUNK_SIZE=1000, CHUNK_OVERLAP=200
```

**문제**: 도서 독후감 특성상 하나의 "개념 단위"가 300~500자 수준임.
청크가 1,000자이면 여러 개념이 뒤섞여 RAGAS가 주장-컨텍스트 매핑을 어렵게 느낄 수 있음.

**개선 방향**:
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100`으로 축소 테스트
- 또는 `RecursiveCharacterTextSplitter` → `MarkdownTextSplitter` 전환 검토 (Notion 마크다운 구조 활용)

**기대 효과**: 청크 내 단일 개념 집중 → Faithfulness 주장 매핑 정확도 향상

---

### 7-3. [단기 / 중간] Top-K 검색 수 조정 실험

**현재**: `TOP_K_RESULTS=3` (서비스), 평가 시 `k=5`

**문제**: 서비스에서 Top-3만 LLM에 전달하면 필요한 청크가 누락될 수 있음.
평가는 k=5로 진행하지만 실제 서비스는 k=3이므로 평가-서비스 간 조건 불일치.

**개선 방향**:
- 서비스도 `TOP_K_RESULTS=5`로 상향하거나
- 평가를 `k=3`으로 맞춰 실제 서비스 조건과 동일하게 측정

---

### 7-4. [중기 / 높음] LLM 모델 업그레이드

**현재**: EXAONE-4.0-1.2B (1.2B 파라미터 소형 모델)

**문제**: 소형 모델 특성상 instruction following이 제한적.
"컨텍스트만 참고"라는 지시를 충실히 따르는 능력이 대형 모델 대비 낮음.

**개선 방향**:
| 후보 모델 | 파라미터 | 특징 |
|----------|---------|------|
| EXAONE-4.0-7.8B | 7.8B | 현재 모델의 업그레이드, 한국어 최적화 |
| Qwen3-8B | 8B | 다국어 강점, instruction following 우수 |
| Llama-3.1-8B | 8B | 범용 고성능 |

**기대 효과**: Faithfulness 70%+ 달성 가능성 높음
**변경 파일**: `.env`의 `OLLAMA_MODEL` 값 수정

---

### 7-5. [중기 / 중간] Reranker 도입

**현재**: hybrid_search 결과를 점수 기준으로 Top-K 직접 사용

**문제**: BM25+벡터 합산 점수가 실제 질문-청크 관련성을 완벽히 반영하지 못할 수 있음.

**개선 방향**:
- `bge-reranker-v2-m3` (이미 `.env`에 `RERANKER_MODEL` 정의됨) 활성화
- hybrid_search Top-10 → Reranker → Top-5 파이프라인 구성

```
hybrid_search(k=10)
    │
    ▼
bge-reranker-v2-m3 (cross-encoder 재정렬)
    │
    ▼
Top-5 최종 컨텍스트 → LLM
```

**기대 효과**: Hit Rate는 이미 100%이나, LLM에 전달되는 청크의 관련성 밀도 향상 → Faithfulness 간접 개선

---

### 7-6. [장기 / 중간] Golden Set 확장 및 다양화

**현재**: 10개 질문, 모두 동일 page_id에서 출처

**문제**:
- 10개 샘플은 통계적으로 신뢰도가 낮음 (95% 신뢰구간 ±15%p 수준)
- 모든 질문이 단일 Notion 페이지에서 나와 다양성 부족

**개선 방향**:
- 30~50개로 확장
- 여러 Notion 페이지/문서 유형 포함 (독후감, 기술 문서, 회의록 등)
- 답변 유형 다양화 (사실 확인형, 요약형, 비교형)

---

## 8. 우선순위 로드맵

```
즉시 (이번 스프린트)
├── 7-1. 프롬프트 강화       → chat_prompt.py 수정, 재측정
└── 7-3. 평가-서비스 k 통일  → TOP_K_RESULTS=5 or 평가 k=3

단기 (2~4주)
├── 7-2. 청크 크기 실험       → 재인덱싱 필요
└── 7-4. 7.8B 모델 테스트    → Ollama 모델 다운로드 후 A/B 비교

중기 (1~2개월)
├── 7-5. Reranker 활성화     → graph 노드 추가
└── 7-6. Golden Set 확장    → 30~50개 질문 확보
```

---

## 9. 평가 재실행 방법

### 환경 설정

```bash
# 별도 venv 필요 (langchain-core 버전 충돌 방지)
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -r requirements-eval.txt
```

### `.env` 필수 설정

```env
ES_HOST=https://es.nabee.ai.kr/
ES_API_KEY=...
OLLAMA_HOST=https://ollama.nabee.ai.kr/
CF_ACCESS_CLIENT_ID=...        # Cloudflare Access
CF_ACCESS_CLIENT_SECRET=...
GROQ_API_KEY=...               # RAGAS 평가 judge (KEY_1)
GROQ_API_KEY_2=...             # RAGAS 평가 judge (KEY_2, 다른 계정 권장)
```

> **주의**: Groq 무료 티어는 100K TPD (일일 토큰) 제한.
> GROQ_API_KEY_2는 **다른 Groq 계정**의 키를 사용해야 실질적인 200K TPD 확보 가능.
> 같은 계정의 두 키는 TPD를 공유함.

### Phase 1: 검색 품질

```bash
python tests/test_search_quality.py --golden-set tests/golden_set.json --k 5
# 결과: tests/eval_result.json
```

### Phase 2: 생성 품질 (RAGAS)

```bash
# 단독 실행 (~5~10분, Ollama 답변 생성 포함)
python tests/test_ragas.py --k 5

# pytest 실행
pytest tests/test_ragas.py -v -s

# 결과: tests/ragas_result.json
```

---

## 10. 트러블슈팅 이력

### 10-1. Groq TPD 초과 시 Faithfulness NaN 급증

**현상**: RAGAS 평가 중 `RateLimitError: Used 99xxx/100000` → 평가 샘플 대부분 NaN

**원인**: Groq 무료 티어 100K TPD 한도. 두 키가 같은 조직이면 TPD 공유.

**해결**: GROQ_API_KEY_2는 **별도 Groq 계정** 키 사용. 당일 TPD 소진 시 약 40분 대기 후 재실행.

---

### 10-2. RAGAS `bool_` JSON 직렬화 오류

**현상**: `TypeError: Object of type bool_ is not JSON serializable`

**원인**: RAGAS 결과의 numpy `bool_` 타입이 Python 기본 `json.dump`에서 직렬화 불가.

**해결**: `bool(faithfulness >= threshold)` 명시적 형변환으로 해결.

---

### 10-3. ragas + langchain-core 버전 충돌

**현상**: 메인 `requirements.txt`에 `langchain-core<0.2.0` 제약 존재. `langchain-groq`는 `>=0.2.0` 요구.

**원인**: 동일 venv에 두 제약 공존 불가.

**해결**: `requirements-eval.txt` + `.venv-eval` 별도 가상환경으로 완전 격리.

---

### 10-4. `LangchainLLMWrapper`, `LangchainEmbeddingsWrapper` Deprecation Warning

**현상**: ragas 0.4.3에서 Langchain 래퍼 deprecated 경고 출력.

**영향**: 기능은 정상 동작. v1.0에서 제거 예정.

**향후 대응**: ragas v1.0 출시 시 `llm_factory`, `embedding_factory` API로 마이그레이션 필요.
