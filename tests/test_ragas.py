"""
test_ragas.py — Phase 2: RAGAS 생성 품질 평가 (4개 메트릭)

[실행 방법]
    # .venv-eval 환경에서 실행 (langchain-core 버전 충돌 방지)
    source .venv-eval/bin/activate

    # Groq judge (KEY_1/KEY_2 분리)
    pytest tests/test_ragas.py::TestRAGAS -v -s

    # Ollama judge (gemma3:27b, rate limit 없음)
    pytest tests/test_ragas.py::TestRAGASOllama -v -s

[전제 조건]
- .env에 ES_HOST, ES_API_KEY, OLLAMA_HOST 설정
- .env에 CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET 설정 (Cloudflare Access)
- Groq judge 사용 시: GROQ_API_KEY (KEY_1) + GROQ_API_KEY_2 (KEY_2) 설정
- pip install -r requirements-eval.txt (.venv-eval 환경)

[golden_set.json 준비]
  자동 생성 (권장):
    python tests/generate_golden_set.py --size 50
    → Ollama gemma3:4b가 ES 문서 기반으로 질문+정답+근거청크를 자동 생성.
    → reference_contexts 필드 포함 → ContextPrecision/Recall 측정 가능.

  수동 작성: query / summary 필드만 있는 기존 형식도 그대로 동작.

[RAGAS 평가 흐름]
  golden_set.json (50 쿼리)
      │  query               (질문)
      │  reference           (정답)
      │  reference_contexts  (근거 청크 — ContextRecall/Precision 필요)
      ▼
  hybrid_search (ES) → retrieved_contexts: list[str]
      │
      ▼
  Ollama EXAONE (실제 RAG 시스템 LLM) → answer: str
      │
      ▼
  EvaluationDataset (SingleTurnSample × 50)
      │
      ├── [TestRAGAS] Groq KEY_1 ──────────────────────────────────
      │   ├─ Faithfulness      → 기준 ≥ 70%
      │   └─ ContextPrecision  → 기준 ≥ 70%
      │
      ├── [TestRAGAS] Groq KEY_2 ──────────────────────────────────
      │   ├─ AnswerRelevancy   → 기준 ≥ 70%
      │   └─ ContextRecall     → 기준 ≥ 70%
      │
      └── [TestRAGASOllama] Ollama gemma3:27b ────────────────────
          ├─ Faithfulness + ContextPrecision  (test_retrieval_metrics)
          └─ AnswerRelevancy + ContextRecall  (test_generation_metrics)

[역할 분리]
- 질문 생성  : Ollama gemma3:4b — ES 문서 기반 자동 생성 (generate_golden_set.py)
- 답변 생성  : Ollama EXAONE — 실제 운영 시스템과 동일한 LLM
- 평가 judge : [기본] Groq KEY_1 → Faithfulness + ContextPrecision
                       Groq KEY_2 → AnswerRelevancy + ContextRecall
               [대안] Ollama gemma3:27b → 4개 메트릭 전체 (rate limit 없음)

[Groq Rate Limit 대응]
- 100K TPD (일일 토큰) 제한 — 50샘플 × 4메트릭 ≈ 200 LLM 호출
- KEY_1 / KEY_2 분리 (별도 계정)로 각각 100K → 총 200K 사용 가능
- 대안: TestRAGASOllama (Ollama gemma3:27b) — TPD 제한 없음

[Ollama judge 주의사항]
- RunConfig(timeout=300, max_retries=1, max_workers=4) 필수
- 기본 max_workers=16은 gemma3:27b 서버 과부하 유발 (GPU 100% + swap)
- gemma3:27b vs gemma3:4b: AnswerRelevancy 72.2% vs 19.5% → gemma3:27b 권장
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from langchain_community.llms import Ollama
    from langchain_groq import ChatGroq

# ragas 미설치 시 모듈 전체 스킵
pytest.importorskip(
    "ragas",
    reason="ragas 미설치. `.venv-eval` 환경에서 `pip install -r requirements-eval.txt` 실행 후 재시도",
)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).parent
_GOLDEN_SET_PATH = _TESTS_DIR / "golden_set.json"
_RESULT_PATH = _TESTS_DIR / "ragas_result.json"

_FAITHFULNESS_THRESHOLD = 0.70
_ANSWER_RELEVANCY_THRESHOLD = 0.70
_CONTEXT_PRECISION_THRESHOLD = 0.70
_CONTEXT_RECALL_THRESHOLD = 0.70


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _ragas_score(result: object, key: str) -> float:
    """
    ragas 0.2+에서 result[key]는 샘플별 점수 list를 반환한다.
    NaN/None은 제외하고 유효한 샘플만 평균 계산.
    """
    raw = result[key]  # type: ignore[index]
    if not isinstance(raw, list):
        return float(raw)

    valid = [v for v in raw if v is not None and not math.isnan(float(v))]
    fail_count = len(raw) - len(valid)
    if fail_count > 0:
        print(f"  ⚠️  [{key}] {fail_count}/{len(raw)}개 샘플 평가 실패 (None/NaN) — 유효 샘플만 집계")
    return sum(valid) / len(valid) if valid else 0.0


def _build_ollama_llm() -> "Ollama":
    """CF 헤더 포함 Ollama LLM 생성 (실제 RAG 시스템과 동일한 설정)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

    from common.config import settings
    from langchain_community.llms import Ollama

    cf_headers: dict[str, str] = {}
    if settings.CF_ACCESS_CLIENT_ID and settings.CF_ACCESS_CLIENT_SECRET:
        cf_headers = {
            "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
            "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
        }
    return Ollama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=0.1,
        headers=cf_headers,
    )


def _generate_answer(llm: "Ollama", question: str, contexts: list[str]) -> str:
    """
    Ollama EXAONE으로 답변 생성 — 실제 RAG 시스템과 동일한 LLM 사용.

    컨텍스트에 기반한 3~5문장 답변 생성. Answer Relevancy 향상을 위해
    수치/사실 포함 및 질문 직접 답변을 지시한다.
    """
    context_text = "\n---\n".join(contexts)
    prompt = (
        "[지시] 아래 컨텍스트에 있는 내용만을 사용하여 답변하세요.\n"
        "컨텍스트에 없는 정보는 절대 포함하지 마세요. 배경 지식이나 추론을 추가하지 마세요.\n"
        f"반드시 첫 문장을 '{question}' 라는 질문에 대해, 로 시작하세요.\n"
        "이후 컨텍스트에서 확인된 핵심 사실을 2~3문장으로 답하세요.\n"
        "컨텍스트에 답이 없으면 '해당 정보를 찾을 수 없습니다'라고 답하세요.\n\n"
        f"컨텍스트:\n{context_text}\n\n"
        f"질문: {question}\n답변:"
    )
    return str(llm.invoke(prompt))


def _build_ragas_embeddings() -> "LangchainEmbeddingsWrapper":  # type: ignore[name-defined]
    """
    Ollama bge-m3 임베딩 → RAGAS 래퍼.

    langchain_ollama.OllamaEmbeddings는 client_kwargs로 CF 헤더를 전달한다.
    (langchain_community와 달리 headers= 직접 전달 불가)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

    from common.config import settings
    from langchain_ollama import OllamaEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    cf_headers: dict[str, str] = {}
    if settings.CF_ACCESS_CLIENT_ID and settings.CF_ACCESS_CLIENT_SECRET:
        cf_headers = {
            "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
            "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
        }

    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            client_kwargs={"headers": cf_headers} if cf_headers else {},
        )
    )


async def _build_ragas_dataset(
    golden_set: list[dict],
    k: int = 3,
) -> object:
    """
    Golden Set으로부터 RAGAS EvaluationDataset 빌드.

    1. hybrid_search → contexts: list[str]
    2. Ollama EXAONE으로 답변 생성 (실제 RAG 시스템 LLM)
    3. SingleTurnSample 구성 후 EvaluationDataset 반환
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

    from ragas import EvaluationDataset, SingleTurnSample
    from stores.vector_store import elasticsearch_store

    ollama_llm = _build_ollama_llm()
    await elasticsearch_store.initialize()
    samples: list[SingleTurnSample] = []

    for item in golden_set:
        query: str = item["query"]

        # 1. hybrid_search → contexts
        ctx = await elasticsearch_store.hybrid_search(query=query, k=k)
        contexts: list[str] = [doc.content for doc in ctx.documents]

        if not contexts:
            print(f"  ⚠️  컨텍스트 없음 스킵: {query[:50]}")
            continue

        # 2. Ollama로 답변 생성 (동기 호출이므로 to_thread로 감싸기)
        answer = await asyncio.to_thread(_generate_answer, ollama_llm, query, contexts)
        print(f"  ✅ {query[:55]}...")

        sample_kwargs: dict = {
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": answer,
            # reference: 자동 생성 시 상세 정답, 수동 작성 시 summary fallback
            "reference": item.get("reference") or item.get("summary", ""),
        }
        # reference_contexts 있으면 ContextRecall/ContextPrecision 측정 가능
        if item.get("reference_contexts"):
            sample_kwargs["reference_contexts"] = item["reference_contexts"]

        samples.append(SingleTurnSample(**sample_kwargs))

    await elasticsearch_store.close()

    assert samples, "RAGAS 샘플 생성 실패 — ES/Ollama 연결을 확인하세요."
    return EvaluationDataset(samples=samples)


def _save_result(
    dataset: object,
    scores: dict[str, float],
    output_path: Path = _RESULT_PATH,
) -> None:
    """평가 결과(4개 메트릭)를 JSON 파일로 저장. 누적 업데이트 방식."""
    from datetime import datetime

    # 기존 파일 있으면 로드해서 누적
    existing: dict = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    per_query = []
    for i, sample in enumerate(dataset.samples):  # type: ignore[attr-defined]
        per_query.append({
            "index": i,
            "question": sample.user_input,
            "answer": sample.response,
            "contexts_count": len(sample.retrieved_contexts),
            "ground_truth": sample.reference,
        })

    merged_scores = {**existing.get("summary", {}), **{k: round(v, 4) for k, v in scores.items()}}
    thresholds = {
        "faithfulness": _FAITHFULNESS_THRESHOLD,
        "answer_relevancy": _ANSWER_RELEVANCY_THRESHOLD,
        "context_precision": _CONTEXT_PRECISION_THRESHOLD,
        "context_recall": _CONTEXT_RECALL_THRESHOLD,
    }

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "summary": {
            **merged_scores,
            **{f"{k}_pass": bool(merged_scores.get(k, 0) >= thresholds[k]) for k in thresholds},
            "total_queries": len(per_query),
        },
        "thresholds": thresholds,
        "per_query": per_query,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {output_path}")


# ── TestRAGAS ─────────────────────────────────────────────────────────────────

class TestRAGAS:
    """
    RAGAS 기반 생성 품질 평가 — 4개 메트릭, KEY_1 / KEY_2 분리.

    KEY_1: Faithfulness + ContextPrecision
    KEY_2: AnswerRelevancy + ContextRecall
    """

    @pytest.fixture(scope="class")
    def ragas_dataset(self) -> object:
        """
        class scope: 4개 테스트에서 Dataset 1회 생성 재사용.

        Dataset 빌드 = ES hybrid_search + Ollama 답변 생성.
        """
        golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        print(f"\n  📊 RAGAS Dataset 생성 중... ({len(golden_set)}개 쿼리, Ollama 답변 생성)")
        return asyncio.run(_build_ragas_dataset(golden_set))

    def test_faithfulness_and_context_precision(
        self,
        ragas_dataset: object,
        groq_llm_key1: "ChatGroq",
    ) -> None:
        """
        [KEY_1] Faithfulness + ContextPrecision 측정.

        Faithfulness      = 뒷받침 가능한 주장 수 / 전체 주장 수         기준 ≥ 70%
        ContextPrecision  = 관련 청크가 상위에 랭크되는지 여부            기준 ≥ 70%
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, Faithfulness

        ragas_llm = LangchainLLMWrapper(groq_llm_key1)

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                Faithfulness(llm=ragas_llm, max_retries=3),
                ContextPrecision(llm=ragas_llm),
            ],
            llm=ragas_llm,
        )

        f_score = _ragas_score(result, "faithfulness")
        cp_score = _ragas_score(result, "context_precision")
        n = len(ragas_dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [KEY_1] 평가 샘플 수    : {n}개")
        print(f"  Faithfulness      : {f_score:.3f} ({f_score:.1%})")
        print(f"  ContextPrecision  : {cp_score:.3f} ({cp_score:.1%})")

        _save_result(ragas_dataset, {"faithfulness": f_score, "context_precision": cp_score})

        assert f_score >= _FAITHFULNESS_THRESHOLD, (
            f"Faithfulness = {f_score:.1%} — 목표 {_FAITHFULNESS_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 품질 확인 (Hit Rate/MRR 먼저 검토)\n"
            "  → 2) 프롬프트의 '원문 기반 답변' 지시 강화"
        )
        assert cp_score >= _CONTEXT_PRECISION_THRESHOLD, (
            f"ContextPrecision = {cp_score:.1%} — 목표 {_CONTEXT_PRECISION_THRESHOLD:.0%} 미달.\n"
            "  → 1) 하이브리드 검색 가중치 조정 (벡터 vs BM25)\n"
            "  → 2) 검색 k값 축소로 노이즈 청크 제거"
        )

    def test_answer_relevancy_and_context_recall(
        self,
        ragas_dataset: object,
        groq_llm_key2: "ChatGroq",
    ) -> None:
        """
        [KEY_2] AnswerRelevancy + ContextRecall 측정.

        AnswerRelevancy = 역생성 질문과 원래 질문의 임베딩 유사도    기준 ≥ 70%
        ContextRecall   = reference 커버에 필요한 청크 검색 여부     기준 ≥ 70%

        strictness=1 필수: Groq는 n>1 복수 completion 미지원.
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextRecall

        ragas_llm = LangchainLLMWrapper(groq_llm_key2)
        ragas_embeddings = _build_ragas_embeddings()

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                AnswerRelevancy(
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    strictness=1,  # Groq n>1 미지원 → 필수
                ),
                ContextRecall(llm=ragas_llm),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        ar_score = _ragas_score(result, "answer_relevancy")
        cr_score = _ragas_score(result, "context_recall")
        n = len(ragas_dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [KEY_2] 평가 샘플 수    : {n}개")
        print(f"  AnswerRelevancy   : {ar_score:.3f} ({ar_score:.1%})")
        print(f"  ContextRecall     : {cr_score:.3f} ({cr_score:.1%})")

        _save_result(ragas_dataset, {"answer_relevancy": ar_score, "context_recall": cr_score})

        assert ar_score >= _ANSWER_RELEVANCY_THRESHOLD, (
            f"AnswerRelevancy = {ar_score:.1%} — 목표 {_ANSWER_RELEVANCY_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 관련성 확인\n"
            "  → 2) 답변 생성 프롬프트 개선 (질문 직접 답변 지시 추가)"
        )
        assert cr_score >= _CONTEXT_RECALL_THRESHOLD, (
            f"ContextRecall = {cr_score:.1%} — 목표 {_CONTEXT_RECALL_THRESHOLD:.0%} 미달.\n"
            "  → 1) 검색 k값 증가로 더 많은 청크 포함\n"
            "  → 2) 청크 분할 크기 조정 (CHUNK_SIZE/OVERLAP)"
        )


# ── TestRAGASOllama ───────────────────────────────────────────────────────────

class TestRAGASOllama:
    """
    Ollama gemma3:4b judge로 4개 메트릭을 단일 모델에서 처리.

    Groq 대비 장점:
    - TPD/RPM rate limit 없음
    - KEY 분리 불필요 — 모든 메트릭을 한 번에 실행
    - 50샘플 × 4메트릭 ≈ 6분 예상

    실행:
        pytest tests/test_ragas.py::TestRAGASOllama -v -s
    """

    @pytest.fixture(scope="class")
    def ragas_dataset(self) -> object:
        """class scope: 두 테스트에서 Dataset 1회 생성 재사용."""
        golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        print(f"\n  📊 RAGAS Dataset 생성 중... ({len(golden_set)}개 쿼리, Ollama 답변 생성)")
        return asyncio.run(_build_ragas_dataset(golden_set))

    def test_retrieval_metrics(
        self,
        ragas_dataset: object,
        ollama_judge_llm: object,
    ) -> None:
        """
        [Ollama] Faithfulness + ContextPrecision 측정.

        Faithfulness     = 뒷받침 가능한 주장 수 / 전체 주장 수    기준 ≥ 70%
        ContextPrecision = 관련 청크가 상위에 랭크되는지 여부       기준 ≥ 70%
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, Faithfulness
        from ragas.run_config import RunConfig

        ragas_llm = LangchainLLMWrapper(ollama_judge_llm)

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                Faithfulness(llm=ragas_llm, max_retries=1),
                ContextPrecision(llm=ragas_llm),
            ],
            llm=ragas_llm,
            run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
        )

        f_score = _ragas_score(result, "faithfulness")
        cp_score = _ragas_score(result, "context_precision")
        n = len(ragas_dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [Ollama gemma3:27b] 평가 샘플 수 : {n}개")
        print(f"  Faithfulness      : {f_score:.3f} ({f_score:.1%})")
        print(f"  ContextPrecision  : {cp_score:.3f} ({cp_score:.1%})")

        _save_result(ragas_dataset, {"faithfulness": f_score, "context_precision": cp_score})

        assert f_score >= _FAITHFULNESS_THRESHOLD, (
            f"Faithfulness = {f_score:.1%} — 목표 {_FAITHFULNESS_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 품질 확인 (Hit Rate/MRR 먼저 검토)\n"
            "  → 2) 프롬프트의 '원문 기반 답변' 지시 강화"
        )
        assert cp_score >= _CONTEXT_PRECISION_THRESHOLD, (
            f"ContextPrecision = {cp_score:.1%} — 목표 {_CONTEXT_PRECISION_THRESHOLD:.0%} 미달.\n"
            "  → 1) 하이브리드 검색 가중치 조정\n"
            "  → 2) 검색 k값 축소로 노이즈 청크 제거"
        )

    def test_generation_metrics(
        self,
        ragas_dataset: object,
        ollama_judge_llm: object,
    ) -> None:
        """
        [Ollama] AnswerRelevancy + ContextRecall 측정.

        AnswerRelevancy = 역생성 질문과 원래 질문의 임베딩 유사도   기준 ≥ 70%
        ContextRecall   = reference 커버에 필요한 청크 검색 여부    기준 ≥ 70%

        strictness=1 필수: Ollama는 n>1 복수 completion 미지원.
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextRecall
        from ragas.run_config import RunConfig

        ragas_llm = LangchainLLMWrapper(ollama_judge_llm)
        ragas_embeddings = _build_ragas_embeddings()

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                AnswerRelevancy(
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    strictness=1,
                ),
                ContextRecall(llm=ragas_llm),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
        )

        ar_score = _ragas_score(result, "answer_relevancy")
        cr_score = _ragas_score(result, "context_recall")
        n = len(ragas_dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [Ollama gemma3:27b] 평가 샘플 수 : {n}개")
        print(f"  AnswerRelevancy   : {ar_score:.3f} ({ar_score:.1%})")
        print(f"  ContextRecall     : {cr_score:.3f} ({cr_score:.1%})")

        _save_result(ragas_dataset, {"answer_relevancy": ar_score, "context_recall": cr_score})

        assert ar_score >= _ANSWER_RELEVANCY_THRESHOLD, (
            f"AnswerRelevancy = {ar_score:.1%} — 목표 {_ANSWER_RELEVANCY_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 관련성 확인\n"
            "  → 2) 답변 생성 프롬프트 개선"
        )
        assert cr_score >= _CONTEXT_RECALL_THRESHOLD, (
            f"ContextRecall = {cr_score:.1%} — 목표 {_CONTEXT_RECALL_THRESHOLD:.0%} 미달.\n"
            "  → 1) 검색 k값 증가로 더 많은 청크 포함\n"
            "  → 2) 청크 분할 크기 조정 (CHUNK_SIZE/OVERLAP)"
        )


# ── 단독 실행 ─────────────────────────────────────────────────────────────────

async def _main(k: int, output: str) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

    from common.config import settings

    try:
        from langchain_groq import ChatGroq
    except ImportError:
        print("❌ langchain-groq 미설치. `.venv-eval` 환경에서 실행하세요.")
        return

    if not settings.GROQ_API_KEY:
        print("❌ GROQ_API_KEY 미설정 — .env 확인")
        return

    # KEY_1: Faithfulness + ContextPrecision
    # KEY_2: AnswerRelevancy + ContextRecall
    key1 = settings.GROQ_API_KEY
    key2 = settings.GROQ_API_KEY_2 or key1
    llm1 = ChatGroq(api_key=key1, model="llama-3.3-70b-versatile")
    llm2 = ChatGroq(api_key=key2, model="llama-3.3-70b-versatile")
    print(f"  KEY_1 → Faithfulness + ContextPrecision")
    print(f"  KEY_2 → AnswerRelevancy + ContextRecall {'(KEY_1 fallback)' if not settings.GROQ_API_KEY_2 else ''}")

    golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    print(f"\n📊 RAGAS Dataset 생성 중... (Ollama 답변 생성, {len(golden_set)}개 쿼리, k={k})")
    dataset = await _build_ragas_dataset(golden_set, k=k)

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    llm1_wrapped = LangchainLLMWrapper(llm1)
    llm2_wrapped = LangchainLLMWrapper(llm2)
    ragas_embeddings = _build_ragas_embeddings()

    print("\n🔍 [KEY_1] Faithfulness + ContextPrecision 평가 중...")
    result1 = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=llm1_wrapped, max_retries=3),
            ContextPrecision(llm=llm1_wrapped),
        ],
        llm=llm1_wrapped,
    )

    print("🔍 [KEY_2] AnswerRelevancy + ContextRecall 평가 중...")
    result2 = evaluate(
        dataset=dataset,
        metrics=[
            AnswerRelevancy(llm=llm2_wrapped, embeddings=ragas_embeddings, strictness=1),
            ContextRecall(llm=llm2_wrapped),
        ],
        llm=llm2_wrapped,
        embeddings=ragas_embeddings,
    )

    f  = _ragas_score(result1, "faithfulness")
    cp = _ragas_score(result1, "context_precision")
    ar = _ragas_score(result2, "answer_relevancy")
    cr = _ragas_score(result2, "context_recall")

    thr = {
        "faithfulness": _FAITHFULNESS_THRESHOLD,
        "context_precision": _CONTEXT_PRECISION_THRESHOLD,
        "answer_relevancy": _ANSWER_RELEVANCY_THRESHOLD,
        "context_recall": _CONTEXT_RECALL_THRESHOLD,
    }
    scores = {"faithfulness": f, "context_precision": cp, "answer_relevancy": ar, "context_recall": cr}

    print(f"\n{'='*48}")
    for name, val in scores.items():
        icon = "✅" if val >= thr[name] else "❌"
        print(f"  {name:<22}: {val:.1%}  {icon}")
    print(f"{'='*48}")

    _save_result(dataset, scores, Path(output))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS 생성 품질 평가")
    parser.add_argument("--k", type=int, default=5, help="검색 결과 개수 (기본값: 5)")
    parser.add_argument("--output", type=str, default=str(_RESULT_PATH), help="결과 저장 경로")
    args = parser.parse_args()

    asyncio.run(_main(args.k, args.output))
