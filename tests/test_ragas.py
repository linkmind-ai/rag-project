"""
test_ragas.py — Phase 2: RAGAS 생성 품질 평가

[실행 방법]
    # .venv-eval 환경에서 실행 (langchain-core 버전 충돌 방지)
    source .venv-eval/bin/activate
    pytest tests/test_ragas.py -v -s

[전제 조건]
- .env에 GROQ_API_KEY 설정 완료
- .env에 ES_HOST, ES_API_KEY, OLLAMA_HOST 설정 완료
- pip install -r requirements-eval.txt (별도 venv 필요)

[RAGAS 평가 흐름]
  golden_set.json (10 쿼리)
      │
      ▼
  hybrid_search (ES) → contexts: list[str]
      │
      ▼
  Ollama EXAONE (실제 RAG 시스템 LLM) → answer: str
      │
      ▼
  EvaluationDataset (SingleTurnSample × 10)
      │
      ▼
  RAGAS evaluate() — 평가 judge: Groq llama-3.3-70b
      ├─ Faithfulness       → 기준 ≥ 70%
      └─ AnswerRelevancy    → 기준 ≥ 70%

[역할 분리]
- 답변 생성: Ollama EXAONE — 실제 운영 시스템과 동일한 LLM으로 측정
- RAGAS 평가(judge): Groq llama-3.3-70b — 답변이 컨텍스트에 근거하는지, 질문에 맞는지 판단

[Groq Rate Limit 대응]
- 100K TPD (일일 토큰) 제한
- KEY_1(GROQ_API_KEY)과 KEY_2(GROQ_API_KEY_2)가 다른 계정이면 각각 100K 확보 가능
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
        "다음 컨텍스트만 참고하여 질문에 답하세요.\n"
        "수치나 구체적 사실을 포함하여 3~5문장으로 작성하세요.\n"
        "컨텍스트에 없는 내용은 답변하지 마세요.\n\n"
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
    k: int = 5,
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

        samples.append(
            SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=answer,
                reference=item.get("summary", ""),
            )
        )

    await elasticsearch_store.close()

    assert samples, "RAGAS 샘플 생성 실패 — ES/Ollama 연결을 확인하세요."
    return EvaluationDataset(samples=samples)


def _save_result(
    result: object,
    dataset: object,
    faithfulness: float,
    answer_relevancy: float,
    output_path: Path = _RESULT_PATH,
) -> None:
    """평가 결과를 JSON 파일로 저장."""
    from datetime import datetime

    per_query = []
    for i, sample in enumerate(dataset.samples):  # type: ignore[attr-defined]
        per_query.append({
            "index": i,
            "question": sample.user_input,
            "answer": sample.response,
            "contexts_count": len(sample.retrieved_contexts),
            "ground_truth": sample.reference,
        })

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "summary": {
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "faithfulness_pass": bool(faithfulness >= _FAITHFULNESS_THRESHOLD),
            "answer_relevancy_pass": bool(answer_relevancy >= _ANSWER_RELEVANCY_THRESHOLD),
            "total_queries": len(per_query),
        },
        "thresholds": {
            "faithfulness": _FAITHFULNESS_THRESHOLD,
            "answer_relevancy": _ANSWER_RELEVANCY_THRESHOLD,
        },
        "per_query": per_query,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {output_path}")


# ── TestRAGAS ─────────────────────────────────────────────────────────────────

class TestRAGAS:
    """RAGAS 기반 생성 품질 평가 — Groq LLM 필요."""

    @pytest.fixture(scope="class")
    def ragas_dataset(self) -> object:
        """
        class scope: Faithfulness + AnswerRelevancy 두 테스트에서 Dataset 1회 생성 재사용.

        Dataset 빌드 = ES hybrid_search + Ollama 답변 생성.
        """
        golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        print(f"\n  📊 RAGAS Dataset 생성 중... ({len(golden_set)}개 쿼리, Ollama 답변 생성)")
        return asyncio.run(_build_ragas_dataset(golden_set))

    def test_faithfulness(
        self,
        ragas_dataset: object,
        groq_eval_llm: ChatGroq,
    ) -> None:
        """
        Faithfulness 측정 — LLM 답변이 검색된 원문 청크에만 근거하는지 검증.

        Faithfulness = 뒷받침 가능한 주장 수 / 전체 주장 수
        기준: ≥ 70%
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        ragas_llm = LangchainLLMWrapper(groq_eval_llm)

        # 메트릭에 llm 직접 전달 필수 (미전달 시 NaN 반환 — ragas 0.2+ 버그)
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[Faithfulness(llm=ragas_llm, max_retries=3)],
            llm=ragas_llm,
        )

        score = _ragas_score(result, "faithfulness")
        print(f"\n  Faithfulness : {score:.3f} ({score:.1%})")
        print(f"  평가 샘플 수 : {len(ragas_dataset.samples)}개")  # type: ignore[attr-defined]

        assert score >= _FAITHFULNESS_THRESHOLD, (
            f"Faithfulness = {score:.1%} — 목표 {_FAITHFULNESS_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 품질 확인 (Hit Rate/MRR 먼저 검토)\n"
            "  → 2) 프롬프트의 '원문 기반 답변' 지시 강화"
        )

    def test_answer_relevancy(
        self,
        ragas_dataset: object,
        groq_eval_llm: ChatGroq,
    ) -> None:
        """
        Answer Relevancy 측정 — 답변이 질문에 관련 있는지 검증.

        역생성된 질문과 원래 질문의 코사인 유사도 (bge-m3 임베딩 사용).
        기준: ≥ 70%

        strictness=1 필수: Groq는 n>1 복수 completion 미지원 (BadRequestError 방지).
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy

        ragas_llm = LangchainLLMWrapper(groq_eval_llm)
        ragas_embeddings = _build_ragas_embeddings()

        result = evaluate(
            dataset=ragas_dataset,
            metrics=[
                AnswerRelevancy(
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    strictness=1,  # Groq n>1 미지원 → 필수
                )
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        score = _ragas_score(result, "answer_relevancy")
        print(f"\n  Answer Relevancy : {score:.3f} ({score:.1%})")
        print(f"  평가 샘플 수     : {len(ragas_dataset.samples)}개")  # type: ignore[attr-defined]

        # 두 메트릭 모두 측정된 시점에 결과 저장
        faithfulness = _ragas_score(result, "faithfulness") if "faithfulness" in result else 0.0  # type: ignore[operator]
        _save_result(result, ragas_dataset, faithfulness, score)

        assert score >= _ANSWER_RELEVANCY_THRESHOLD, (
            f"Answer Relevancy = {score:.1%} — 목표 {_ANSWER_RELEVANCY_THRESHOLD:.0%} 미달.\n"
            "  → 1) RAG 컨텍스트 관련성 확인\n"
            "  → 2) 답변 생성 프롬프트 개선 (질문 직접 답변 지시 추가)"
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

    # Groq는 RAGAS 평가(judge) 전용 — 답변 생성은 Ollama가 담당
    eval_key = settings.GROQ_API_KEY_2 or settings.GROQ_API_KEY
    eval_llm = ChatGroq(api_key=eval_key, model="llama-3.3-70b-versatile")
    if settings.GROQ_API_KEY_2:
        print("  ✅ GROQ KEY_2 사용 (평가 judge)")
    else:
        print("  ⚠️  GROQ_API_KEY_2 미설정 — KEY_1으로 평가")

    golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    print(f"📊 RAGAS Dataset 생성 중... (Ollama 답변 생성, {len(golden_set)}개 쿼리, k={k})")
    dataset = await _build_ragas_dataset(golden_set, k=k)

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, AnswerRelevancy

    ragas_llm = LangchainLLMWrapper(eval_llm)
    ragas_embeddings = _build_ragas_embeddings()

    print("\n🔍 RAGAS 평가 실행 중...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm, max_retries=3),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    faithfulness = _ragas_score(result, "faithfulness")
    answer_relevancy = _ragas_score(result, "answer_relevancy")

    print(f"\n{'='*40}")
    print(f"  Faithfulness     : {faithfulness:.1%}  {'✅' if faithfulness >= _FAITHFULNESS_THRESHOLD else '❌'}")
    print(f"  Answer Relevancy : {answer_relevancy:.1%}  {'✅' if answer_relevancy >= _ANSWER_RELEVANCY_THRESHOLD else '❌'}")
    print(f"{'='*40}")

    _save_result(result, dataset, faithfulness, answer_relevancy, Path(output))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS 생성 품질 평가")
    parser.add_argument("--k", type=int, default=5, help="검색 결과 개수 (기본값: 5)")
    parser.add_argument("--output", type=str, default=str(_RESULT_PATH), help="결과 저장 경로")
    args = parser.parse_args()

    asyncio.run(_main(args.k, args.output))
