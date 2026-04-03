"""
test_e2e.py — 전체 RAG 파이프라인 E2E RAGAS 평가 (테스트 클래스)

[test_ragas.py vs test_ragas_e2e.py 비교]

  test_ragas.py (기존)           test_ragas_e2e/test_e2e.py (이 파일)
  ─────────────────────────────  ────────────────────────────────────────
  ES store 직접 호출             RAGService.process_query()
  하드코딩 테스트 프롬프트        apps/prompts/chat_prompt.py (운영 프롬프트)
  HyDE / grade / evidence 없음   N1~N7 전 노드 실행

[실행]
  pytest tests/ragas_e2e/test_e2e.py::TestRAGASE2EQwen25 -v -s
  pytest tests/ragas_e2e/test_e2e.py::TestRAGASE2EQwen25 -v -s --golden-set tests/golden_sets/golden_set_138.json

[단독 실행 (PyCharm Run Configuration)]
  python tests/ragas_e2e/test_e2e.py --judge qwen25
  python tests/ragas_e2e/test_e2e.py --judge qwen25 --golden-set tests/golden_sets/golden_set_138.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# 단독 실행(PyCharm debug) 시 tests/ 와 apps/ 를 sys.path에 추가.
# pytest 실행 시에는 conftest.py가 이미 처리하므로 조건 분기로 중복 방지.
_TESTS_DIR = Path(__file__).parent.parent
_APPS_DIR = _TESTS_DIR.parent / "apps"
for _p in (_TESTS_DIR, _APPS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pytest

pytest.importorskip(
    "ragas",
    reason="ragas 미설치. `.venv-eval` 환경에서 `pip install -r requirements-eval.txt` 실행",
)

from ragas_e2e._helpers import ANSWER_RELEVANCY_THRESHOLD  # noqa: E402
from ragas_e2e._helpers import (CONTEXT_PRECISION_THRESHOLD,
                                CONTEXT_RECALL_THRESHOLD,
                                FAITHFULNESS_THRESHOLD, build_ragas_embeddings,
                                ragas_score, resolve_golden_set,
                                save_e2e_result)
from ragas_e2e._pipeline import build_e2e_dataset  # noqa: E402

# ── TestRAGASE2EQwen25 ────────────────────────────────────────────────────────


class TestRAGASE2EQwen25:
    """
    E2E RAGAS 평가 — qwen2.5:72b judge (표준).

    refactoring_log.md (2026-03-21): 4개 지표 모두 threshold 통과 확인.
    AnswerRelevancy 70.8% (gemma3:27b 68.8% 미달 대비 개선).
    """

    @pytest.fixture(scope="class")
    def qwen25_judge_llm(self) -> object:
        """qwen2.5:72b — non-reasoning, 한국어 특화."""
        from common.config import settings
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model="qwen2.5:72b",
            temperature=0,
            num_predict=512,
            client_kwargs={
                "headers": {
                    "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
                    "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
                }
            },
        )

    @pytest.fixture(scope="class")
    def e2e_dataset(
        self, request: pytest.FixtureRequest
    ) -> tuple[object, list[dict[str, Any]]]:
        """class scope: 두 테스트 메서드에서 파이프라인 결과 재사용.
        --golden-set CLI 옵션으로 파일 지정 가능 (미지정 시 golden_sets/golden_set_100.json).
        """
        path = resolve_golden_set(request.config.getoption("--golden-set"))
        golden_set = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"\n  E2E 파이프라인 시작... ({len(golden_set)}개 쿼리, {path.name}, 전체 7노드)"
        )
        return asyncio.run(build_e2e_dataset(golden_set))

    def test_retrieval_metrics(
        self,
        e2e_dataset: tuple[object, list[dict[str, Any]]],
        qwen25_judge_llm: object,
    ) -> None:
        """
        Faithfulness + ContextPrecision.

        미달 시 E2E 관점 개선 방향:
          Faithfulness    → N6 chat_prompt.py 지시 강화 / N3 grade 기준 완화
          ContextPrecision → N1 HyDE 프롬프트 개선 / N2 하이브리드 가중치 조정
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, Faithfulness
        from ragas.run_config import RunConfig

        dataset, diagnostics = e2e_dataset
        # --- BREAKPOINT: dataset.samples[0]으로 첫 번째 샘플 확인 ---
        ragas_llm = LangchainLLMWrapper(qwen25_judge_llm)

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=ragas_llm, max_retries=1),
                ContextPrecision(llm=ragas_llm),
            ],
            llm=ragas_llm,
            run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
        )

        # --- BREAKPOINT: result로 샘플별 점수 확인 ---
        f_score = ragas_score(result, "faithfulness")
        cp_score = ragas_score(result, "context_precision")
        n = len(dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [E2E / qwen2.5:72b] 샘플: {n}개")
        print(f"  Faithfulness      : {f_score:.3f} ({f_score:.1%})")
        print(f"  ContextPrecision  : {cp_score:.3f} ({cp_score:.1%})")

        save_e2e_result(
            dataset,
            {"faithfulness": f_score, "context_precision": cp_score},
            diagnostics,
        )

        assert f_score >= FAITHFULNESS_THRESHOLD, (
            f"Faithfulness {f_score:.1%} < {FAITHFULNESS_THRESHOLD:.0%}\n"
            "  → N6 chat_prompt.py 지시 강화 / N3 grade 기준 조정"
        )
        assert cp_score >= CONTEXT_PRECISION_THRESHOLD, (
            f"ContextPrecision {cp_score:.1%} < {CONTEXT_PRECISION_THRESHOLD:.0%}\n"
            "  → N1 HyDE 프롬프트 개선 / N2 하이브리드 가중치 조정"
        )

    def test_generation_metrics(
        self,
        e2e_dataset: tuple[object, list[dict[str, Any]]],
        qwen25_judge_llm: object,
    ) -> None:
        """
        AnswerRelevancy + ContextRecall.

        미달 시 E2E 관점 개선 방향:
          AnswerRelevancy → chat_prompt.py에 질문 재인용 지시 추가 (refactoring_log 방향 B)
          ContextRecall   → TOP_K_RESULTS 증가 / N3 grade 기준 완화

        strictness=1: Ollama는 n>1 복수 completion 미지원.
        """
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextRecall
        from ragas.run_config import RunConfig

        dataset, diagnostics = e2e_dataset
        # --- BREAKPOINT: dataset.samples[0].response로 생성된 답변 확인 ---
        ragas_llm = LangchainLLMWrapper(qwen25_judge_llm)
        ragas_embeddings = build_ragas_embeddings()

        result = evaluate(
            dataset=dataset,
            metrics=[
                AnswerRelevancy(
                    llm=ragas_llm, embeddings=ragas_embeddings, strictness=1
                ),
                ContextRecall(llm=ragas_llm),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
        )

        # --- BREAKPOINT: result로 샘플별 AnswerRelevancy 확인 ---
        ar_score = ragas_score(result, "answer_relevancy")
        cr_score = ragas_score(result, "context_recall")
        n = len(dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [E2E / qwen2.5:72b] 샘플: {n}개")
        print(f"  AnswerRelevancy   : {ar_score:.3f} ({ar_score:.1%})")
        print(f"  ContextRecall     : {cr_score:.3f} ({cr_score:.1%})")

        save_e2e_result(
            dataset,
            {"answer_relevancy": ar_score, "context_recall": cr_score},
            diagnostics,
        )

        assert ar_score >= ANSWER_RELEVANCY_THRESHOLD, (
            f"AnswerRelevancy {ar_score:.1%} < {ANSWER_RELEVANCY_THRESHOLD:.0%}\n"
            "  → chat_prompt.py에 질문 재인용 지시 추가 (refactoring_log 방향 B)"
        )
        assert cr_score >= CONTEXT_RECALL_THRESHOLD, (
            f"ContextRecall {cr_score:.1%} < {CONTEXT_RECALL_THRESHOLD:.0%}\n"
            "  → TOP_K_RESULTS 증가 / N3 grade 기준 완화"
        )


# ── TestRAGASE2EGroq ──────────────────────────────────────────────────────────


class TestRAGASE2EGroq:
    """
    E2E RAGAS 평가 — Groq llama-3.3-70b judge.

    KEY_1: Faithfulness + ContextPrecision
    KEY_2: AnswerRelevancy + ContextRecall (KEY_2 없으면 KEY_1 fallback)
    """

    @pytest.fixture(scope="class")
    def groq_llm_key1(self) -> object:
        """KEY_1 Groq LLM — Faithfulness + ContextPrecision."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            pytest.skip("langchain-groq 미설치")

        from common.config import settings

        if not settings.GROQ_API_KEY:
            pytest.skip("GROQ_API_KEY 미설정")
        return ChatGroq(api_key=settings.GROQ_API_KEY, model="llama-3.3-70b-versatile")

    @pytest.fixture(scope="class")
    def groq_llm_key2(self) -> object:
        """KEY_2 Groq LLM — AnswerRelevancy + ContextRecall."""
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            pytest.skip("langchain-groq 미설치")

        from common.config import settings

        key = settings.GROQ_API_KEY_2 or settings.GROQ_API_KEY
        if not key:
            pytest.skip("GROQ_API_KEY 미설정")
        return ChatGroq(api_key=key, model="llama-3.3-70b-versatile")

    @pytest.fixture(scope="class")
    def e2e_dataset(
        self, request: pytest.FixtureRequest
    ) -> tuple[object, list[dict[str, Any]]]:
        """class scope: 두 테스트 메서드에서 파이프라인 결과 재사용.
        --golden-set CLI 옵션으로 파일 지정 가능 (미지정 시 golden_sets/golden_set_100.json).
        """
        path = resolve_golden_set(request.config.getoption("--golden-set"))
        golden_set = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"\n  E2E 파이프라인 시작... ({len(golden_set)}개 쿼리, {path.name}, 전체 7노드)"
        )
        return asyncio.run(build_e2e_dataset(golden_set))

    def test_faithfulness_and_context_precision(
        self,
        e2e_dataset: tuple[object, list[dict[str, Any]]],
        groq_llm_key1: object,
    ) -> None:
        """[KEY_1] Faithfulness + ContextPrecision."""
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import ContextPrecision, Faithfulness
        from ragas.run_config import RunConfig

        dataset, diagnostics = e2e_dataset
        ragas_llm = LangchainLLMWrapper(groq_llm_key1)

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(llm=ragas_llm, max_retries=3),
                ContextPrecision(llm=ragas_llm),
            ],
            llm=ragas_llm,
            run_config=RunConfig(timeout=300, max_retries=3, max_workers=2),
        )

        # --- BREAKPOINT: 샘플별 Faithfulness 점수 확인 ---
        f_score = ragas_score(result, "faithfulness")
        cp_score = ragas_score(result, "context_precision")
        n = len(dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [E2E / Groq KEY_1] 샘플: {n}개")
        print(f"  Faithfulness      : {f_score:.3f} ({f_score:.1%})")
        print(f"  ContextPrecision  : {cp_score:.3f} ({cp_score:.1%})")

        save_e2e_result(
            dataset,
            {"faithfulness": f_score, "context_precision": cp_score},
            diagnostics,
        )

        assert (
            f_score >= FAITHFULNESS_THRESHOLD
        ), f"Faithfulness {f_score:.1%} < {FAITHFULNESS_THRESHOLD:.0%}"
        assert (
            cp_score >= CONTEXT_PRECISION_THRESHOLD
        ), f"ContextPrecision {cp_score:.1%} < {CONTEXT_PRECISION_THRESHOLD:.0%}"

    def test_answer_relevancy_and_context_recall(
        self,
        e2e_dataset: tuple[object, list[dict[str, Any]]],
        groq_llm_key2: object,
    ) -> None:
        """[KEY_2] AnswerRelevancy + ContextRecall. strictness=1: Groq n>1 미지원."""
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextRecall
        from ragas.run_config import RunConfig

        dataset, diagnostics = e2e_dataset
        ragas_llm = LangchainLLMWrapper(groq_llm_key2)
        ragas_embeddings = build_ragas_embeddings()

        result = evaluate(
            dataset=dataset,
            metrics=[
                AnswerRelevancy(
                    llm=ragas_llm, embeddings=ragas_embeddings, strictness=1
                ),
                ContextRecall(llm=ragas_llm),
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=300, max_retries=3, max_workers=2),
        )

        # --- BREAKPOINT: 샘플별 AnswerRelevancy 점수 확인 ---
        ar_score = ragas_score(result, "answer_relevancy")
        cr_score = ragas_score(result, "context_recall")
        n = len(dataset.samples)  # type: ignore[attr-defined]

        print(f"\n  [E2E / Groq KEY_2] 샘플: {n}개")
        print(f"  AnswerRelevancy   : {ar_score:.3f} ({ar_score:.1%})")
        print(f"  ContextRecall     : {cr_score:.3f} ({cr_score:.1%})")

        save_e2e_result(
            dataset,
            {"answer_relevancy": ar_score, "context_recall": cr_score},
            diagnostics,
        )

        assert (
            ar_score >= ANSWER_RELEVANCY_THRESHOLD
        ), f"AnswerRelevancy {ar_score:.1%} < {ANSWER_RELEVANCY_THRESHOLD:.0%}"
        assert (
            cr_score >= CONTEXT_RECALL_THRESHOLD
        ), f"ContextRecall {cr_score:.1%} < {CONTEXT_RECALL_THRESHOLD:.0%}"


# ── 단독 실행 ─────────────────────────────────────────────────────────────────


def _build_judge_llm(judge: str) -> Any:
    """
    judge 이름으로 LLM 인스턴스 생성.

    Args:
        judge: "qwen25" 또는 "groq"

    Returns:
        ChatOllama 또는 ChatGroq 인스턴스
    """
    from common.config import settings

    if judge == "qwen25":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model="qwen2.5:72b",
            temperature=0,
            num_predict=512,
            client_kwargs={
                "headers": {
                    "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
                    "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
                }
            },
        )

    from langchain_groq import ChatGroq

    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY 미설정")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model="llama-3.3-70b-versatile")


def _print_scores(scores: dict[str, float]) -> None:
    """점수 테이블을 콘솔에 출력."""
    thresholds: dict[str, float] = {
        "faithfulness": FAITHFULNESS_THRESHOLD,
        "context_precision": CONTEXT_PRECISION_THRESHOLD,
        "answer_relevancy": ANSWER_RELEVANCY_THRESHOLD,
        "context_recall": CONTEXT_RECALL_THRESHOLD,
    }
    print(f"\n{'=' * 55}")
    for name, val in scores.items():
        icon = "✅" if val >= thresholds[name] else "❌"
        print(f"  {name:<22}: {val:.1%}  {icon}")
    print(f"{'=' * 55}")


async def _main(judge: str, golden_set_path: str | None = None) -> None:
    """
    pytest 없이 직접 실행 — 전체 파이프라인 + RAGAS 평가.

    PyCharm Run Configuration:
      Script path: tests/ragas_e2e/test_e2e.py
      Parameters : --judge qwen25
    """
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (AnswerRelevancy, ContextPrecision,
                               ContextRecall, Faithfulness)
    from ragas.run_config import RunConfig

    path = resolve_golden_set(golden_set_path)
    golden_set = json.loads(path.read_text(encoding="utf-8"))
    judge_label = "qwen2.5:72b" if judge == "qwen25" else "Groq llama-3.3-70b"
    print(
        f"E2E 파이프라인 시작 ({len(golden_set)}개 쿼리, {path.name}, judge: {judge_label})..."
    )

    dataset, diagnostics = await build_e2e_dataset(golden_set)

    judge_llm = _build_judge_llm(judge)
    print(f"RAGAS 평가 중 (judge: {judge_label})...")
    ragas_llm = LangchainLLMWrapper(judge_llm)
    ragas_embeddings = build_ragas_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=ragas_llm, max_retries=1),
            ContextPrecision(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
            ContextRecall(llm=ragas_llm),
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=300, max_retries=1, max_workers=2),
    )

    # --- BREAKPOINT: result 전체 점수 확인 ---
    scores = {
        "faithfulness": ragas_score(result, "faithfulness"),
        "context_precision": ragas_score(result, "context_precision"),
        "answer_relevancy": ragas_score(result, "answer_relevancy"),
        "context_recall": ragas_score(result, "context_recall"),
    }
    _print_scores(scores)
    save_e2e_result(dataset, scores, diagnostics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E2E RAG 파이프라인 RAGAS 평가")
    parser.add_argument(
        "--judge",
        choices=["qwen25", "groq"],
        default="qwen25",
        help="judge 모델 선택 (기본값: qwen25)",
    )
    parser.add_argument(
        "--golden-set",
        default=None,
        metavar="PATH",
        help="golden_set JSON 파일 경로 (기본값: tests/golden_sets/golden_set_100.json)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.judge, args.golden_set))
