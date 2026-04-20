# RAGAS Evaluation Report

**평가 일시**: 2026-04-14 19:37:38  
**파이프라인**: `RAGService.process_query()` → `RAGGraph (N1~N7)`  
**Judge 모델**: `gpt-4o-mini`  
**평가 데이터셋**: `golden_set_138` (총 138개 쿼리)

---

## 1. 평가 데이터셋

RAG 시스템이 **내부 지식 기반 질의와 open-domain 질의 모두에 대해 안정적으로 동작하는지**를 검증하기 위해 두 가지 유형의 데이터셋을 구성하여 평가를 수행하였습니다.

### 1-1. Notion-based Dataset (Closed-domain)

- 데모용 Notion 페이지를 기반으로 생성된 QA 데이터
- context precision / recall을 측정하기 위해 query - response - reference context 조합으로 데이터 샘플을 구성

**평가 목적:**

- hybrid search 성능 검증
- 내부 문서 기반 답변 품질 검증

---

### 1-2. Open-domain Dataset

- `iamjoon/klue-mrc-ko-rag-dataset` @HuggingFace
- 한국어 MRC 기반 RAG 평가 데이터셋
- 다양한 주제와 질문 유형을 포함

`iamjoon/klue-mrc-ko-rag-dataset`에 포함된 샘플들 중에서 시간에 독립적이고, 명확한 대상과 단일 정답을 가지며, 추가 맥락 없이 context로부터 직접 답할 수 있는 query 38개를 선별해서 평가 데이터셋에 포함시켰습니다.

**평가 목적:**

- 외부 지식이 필요한 상황에서의 대응 능력
- open-domain 질의에 대한 정보 검색 및 답변 생성 성능 확인

---

## 2. 평가 결과 요약

| 지표 | 점수 | 임계값 | 통과 여부 |
|------|------|--------|-----------|
| Faithfulness (충실도) | **0.8719** | 0.70 | ✅ PASS |
| Context Precision (문맥 정밀도) | **0.8901** | 0.70 | ✅ PASS |
| Context Recall (문맥 재현율) | **0.7654** | 0.60 | ✅ PASS |
| Answer Relevancy (답변 관련성) | **0.5402** | 0.60 | ❌ FAIL |

**종합 판정: FAIL** — 4개 지표 중 1개 미달 (Answer Relevancy)

---

## 2. 지표별 분석

### ✅ Faithfulness — 0.8719
생성된 답변이 검색된 문맥(context)에 근거하는 정도. 임계값(0.70) 대비 +0.17 초과 달성.  
온도를 0.7 → 0.2로 낮춘 효과가 직접적으로 반영된 것으로 판단됨. 할루시네이션 감소에 기여.

### ✅ Context Precision — 0.8901
검색된 문서 중 실제로 답변 생성에 유용한 문서 비율. 가장 높은 점수로 grade_documents 노드의 relevancy threshold 튜닝과 prompt 수정의 성과가 명확히 나타남.

### ✅ Context Recall — 0.7654
정답 생성에 필요한 정보가 검색 단계에서 얼마나 포함되었는지. 임계값(0.60) 대비 +0.17 초과. 하이브리드 검색(벡터 + BM25) 파이프라인이 안정적으로 작동 중.

### ❌ Answer Relevancy — 0.5402
생성된 답변이 입력 질문에 얼마나 관련성 있게 답변하는지. 임계값(0.60) 대비 -0.06 미달.  
질문의 의도를 벗어나거나 과도하게 문맥을 나열하는 답변이 점수를 낮춘 것으로 추정됨. **최우선 개선 대상.**

---

## 3. 이번 평가 전 주요 변경사항

| 변경 항목 | 내용 |
|-----------|------|
| Grader 모델 교체 | `exaone-4.0-1.2b` → `gemma3-4b` |
| Generator 온도 조정 | `0.7` → `0.2` |
| Grade Documents 노드 | relevancy threshold 튜닝 |
| 프롬프트 수정 | `grade_document`, `query_rewrite` 프롬프트 개선 |

**효과**: Faithfulness·Context Precision·Context Recall 모두 임계값을 상회하는 결과를 달성. 특히 Context Precision(0.89)은 grader 모델 교체와 threshold 튜닝의 복합 효과로 보임.

---

## 4. Answer Relevancy 개선 방향

Answer Relevancy가 유일하게 임계값을 충족하지 못했으며, 점수 갭(-0.06)은 아래 원인으로 분석됨.

### Root Cause Analysis

> **측정 전제**: RAGAS Answer Relevancy는 생성된 답변으로부터 역방향으로 질문을 생성(question generation)한 뒤, 그 질문들이 원래 질문과 얼마나 유사한지를 임베딩 코사인 유사도로 측정한다. 즉, "답변이 원래 질문을 유발할 만한 내용으로 구성되어 있는가"를 본다. 따라서 점수가 낮다는 것은 **답변의 초점이 질문의 핵심 의도에서 벗어나 있다**는 신호다.

#### 지표 간 교차 분석 (Metric Cross-Analysis)

| 비교 쌍 | 해석 |
|---------|------|
| Faithfulness(0.87) ↑ & Answer Relevancy(0.54) ↓ | 답변이 검색 문서에 충실하게 작성되었음에도 질문 의도와 괴리. "충실하지만 엉뚱한" 답변 패턴 존재 가능성 |
| Context Precision(0.89) ↑ & Answer Relevancy(0.54) ↓ | 유용한 문서를 골라내는 것 자체는 성공. 병목은 검색이 아닌 **생성 단계** |
| Context Recall(0.77) ↑ & Answer Relevancy(0.54) ↓ | 필요한 정보가 컨텍스트에 충분히 포함되어 있음. 즉 재료는 있으나 **조리(문장화) 방식이 문제** |

→ 세 지표의 패턴이 일관되게 가리키는 결론: **생성 단계 프롬프트 및 답변 구성 로직**이 핵심 원인 위치.

---

#### RC-1. Generator 프롬프트의 초점 부재 (추정 기여도: 高)

**증상**: Faithfulness는 높지만 Answer Relevancy가 낮음 → 문서를 잘 인용하되 질문의 핵심에 수렴하지 않는 답변.

**인과 체인 (5 Whys)**:
1. Answer Relevancy가 낮다
2. → 답변에서 역생성된 질문이 원래 질문과 다르다
3. → 답변이 질문의 핵심 키워드·의도보다 검색된 문서 전체를 요약하는 방향으로 작성되었다
4. → 현재 시스템 프롬프트에 "질문에 직접 답하라"는 명시적 포커싱 지시가 없고, 문서 인용 충실성만 강조되어 있다
5. → 온도를 0.7 → 0.2로 낮춘 후 할루시네이션은 줄었으나, LLM이 보수적으로 문서 내용을 나열하는 패턴이 강화됨

**검증 방법**: Answer Relevancy 최하위 샘플(점수 < 0.4)을 추출하여 답변 패턴이 "문서 요약형"인지 "질문 직접 응답형"인지 수동 분류.

---

#### RC-2. Query Rewrite 노드의 의미 변형 (추정 기여도: 中)

**증상**: Rewrite된 쿼리가 검색 성능(Recall 0.77)에는 기여하지만, 원래 질문의 의도를 바꾸어 Generator의 응답 방향이 틀어질 수 있음.

**인과 체인**:
1. Answer Relevancy 측정 시 비교 기준은 **원래(original) 질문**이다
2. → `query_rewrite` 노드가 검색 최적화를 위해 의미를 확장·변형할 경우, Generator는 rewritten query 기준으로 답변을 생성
3. → 생성된 답변이 original 질문이 아닌 rewritten query에 맞춰져 RAGAS 유사도가 낮게 산출됨
4. → 현재 `query_rewrite` 프롬프트에는 "검색 효율 향상" 목적만 명시, 원래 의도 보존 제약이 없음

**검증 방법**: `query_rewrite` 전·후 쿼리 쌍 20~30개를 추출하여 의미 보존율을 임베딩 유사도로 측정 (임계: cosine ≥ 0.85).

---

#### RC-3. Multi-turn 대화 이력의 컨텍스트 오염 (추정 기여도: 低~中)

**증상**: 단일 턴 질의 대비 멀티 턴 세션에서 Answer Relevancy가 더 낮게 관찰될 가능성.

**인과 체인**:
1. `get_recent_messages()`로 반환된 N개의 이전 메시지가 프롬프트에 추가됨
2. → 이전 대화 내용과 관련된 어휘·주제가 LLM의 attention에서 현재 질문보다 높은 가중치를 가질 수 있음
3. → 결과적으로 현재 질문의 의도보다 이전 흐름을 이어가는 답변이 생성됨
4. → 현재 `get_recent_messages` 반환 개수가 고정값이며, 질문 유형(단발성 vs 문맥 의존적)에 따른 동적 조정이 없음

**검증 방법**: 평가 데이터셋을 single-turn / multi-turn으로 구분하여 그룹별 Answer Relevancy 점수를 비교. 그룹 간 격차가 0.05 이상이면 기여 요인으로 확정.

---

#### 원인 우선순위 요약

| 원인 | 영향 범위 | 수정 비용 | 우선순위 |
|------|-----------|-----------|---------|
| RC-1. Generator 프롬프트 초점 부재 | 전체 쿼리 | 낮음 (프롬프트 수정) | **P0** |
| RC-2. Query Rewrite 의미 변형 | rewrite 적용 쿼리 | 중간 (프롬프트 + 테스트) | **P1** |
| RC-3. Multi-turn 이력 오염 | 멀티 턴 세션 | 중간 (파라미터 실험) | **P2** |

### 개선 액션 아이템

- [ ] Generator 시스템 프롬프트에 "질문의 핵심에 직접 답하라" 지시 강화
- [ ] `query_rewrite` 출력의 의미 보존 여부를 검증하는 단위 테스트 추가
- [ ] Answer Relevancy 하위 케이스(예: 점수 < 0.4)를 샘플링하여 실패 패턴 분석
- [ ] 멀티 턴 대화에서 `get_recent_messages` 반환 개수(현재 설정값) 실험적 조정

---

## 5. 결론 및 다음 단계

이번 평가는 grader 모델 교체 및 프롬프트 튜닝을 통해 검색 품질(Context Precision 0.89, Context Recall 0.77)과 생성 충실도(Faithfulness 0.87) 측면에서 유의미한 개선을 확인했습니다.

**단일 미달 지표인 Answer Relevancy(0.54)를 0.60 이상으로 끌어올리는 것이 다음 스프린트의 핵심 목표**이며, Generator 프롬프트 개선과 실패 케이스 샘플 분석을 병행하여 진행합니다.
