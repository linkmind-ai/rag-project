"""
_helpers.py — RAGAS E2E 평가 공통 유틸리티

포함 항목:
  - 경로 상수 / 임계값 상수
  - _ragas_score()         : NaN 제외 유효 샘플 평균
  - _build_ragas_embeddings(): Ollama bge-m3 → RAGAS 래퍼
  - _save_e2e_result()     : 점수 + 파이프라인 진단 JSON 저장
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 경로 상수 ─────────────────────────────────────────────────────────────────

_TESTS_DIR = Path(__file__).parent.parent  # tests/
GOLDEN_SETS_DIR = _TESTS_DIR / "golden_sets"
DEFAULT_GOLDEN_SET = GOLDEN_SETS_DIR / "golden_set_100.json"
_E2E_RESULT_PATH = _TESTS_DIR / "ragas_e2e_result.json"
_E2E_DIAGNOSTICS_PATH = _TESTS_DIR / "ragas_e2e_diagnostics.json"


def resolve_golden_set(path: str | Path | None = None) -> Path:
    """golden_set 파일 경로를 반환. path 미지정 시 DEFAULT_GOLDEN_SET 사용."""
    resolved = Path(path) if path else DEFAULT_GOLDEN_SET
    if not resolved.exists():
        raise FileNotFoundError(f"golden_set 파일을 찾을 수 없습니다: {resolved}")
    return resolved


# ── RAGAS 임계값 ──────────────────────────────────────────────────────────────

FAITHFULNESS_THRESHOLD = 0.70
ANSWER_RELEVANCY_THRESHOLD = 0.70
CONTEXT_PRECISION_THRESHOLD = 0.70
CONTEXT_RECALL_THRESHOLD = 0.70

_THRESHOLDS: dict[str, float] = {
    "faithfulness": FAITHFULNESS_THRESHOLD,
    "answer_relevancy": ANSWER_RELEVANCY_THRESHOLD,
    "context_precision": CONTEXT_PRECISION_THRESHOLD,
    "context_recall": CONTEXT_RECALL_THRESHOLD,
}


# ── RAGAS 점수 계산 ───────────────────────────────────────────────────────────


def ragas_score(result: object, key: str) -> float:
    """NaN/None 제외 유효 샘플의 평균 점수를 반환."""
    raw = result[key]  # type: ignore[index]
    if not isinstance(raw, list):
        return float(raw)

    valid = [v for v in raw if v is not None and not math.isnan(float(v))]
    fail_count = len(raw) - len(valid)
    if fail_count > 0:
        print(f"  ⚠️  [{key}] {fail_count}/{len(raw)}개 NaN — 유효 샘플만 집계")
    return sum(valid) / len(valid) if valid else 0.0


# ── Ollama 임베딩 (AnswerRelevancy 측정용) ────────────────────────────────────


def build_ragas_embeddings() -> object:
    """
    Ollama bge-m3 임베딩 → RAGAS LangchainEmbeddingsWrapper.

    CF 헤더는 langchain_ollama 방식(client_kwargs)으로 전달.
    """
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


# ── 결과 저장 ─────────────────────────────────────────────────────────────────


def save_e2e_result(
    dataset: object,
    scores: dict[str, float],
    diagnostics: list[dict[str, Any]],
    result_path: Path = _E2E_RESULT_PATH,
    diagnostics_path: Path = _E2E_DIAGNOSTICS_PATH,
) -> None:
    """
    RAGAS 점수(ragas_e2e_result.json) + 파이프라인 진단(ragas_e2e_diagnostics.json) 저장.

    기존 점수는 merge하여 누적 (회차별 비교 가능).
    진단 파일에는 HyDE 출력, 웹검색 여부, 근거 인덱스 등 노드별 중간 결과 포함.
    """
    existing: dict = {}
    if result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    merged = {
        **existing.get("summary", {}),
        **{k: round(v, 4) for k, v in scores.items()},
    }

    result_out = {
        "evaluated_at": datetime.now().isoformat(),
        "pipeline": "RAGService.process_query() → RAGGraph (N1~N7)",
        "summary": {
            **merged,
            **{
                f"{k}_pass": bool(merged.get(k, 0) >= thr)
                for k, thr in _THRESHOLDS.items()
            },
            "total_queries": len(dataset.samples),  # type: ignore[attr-defined]
        },
        "thresholds": _THRESHOLDS,
    }
    result_path.write_text(
        json.dumps(result_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    web_count = sum(1 for d in diagnostics if d["web_search_triggered"])
    diag_out = {
        "saved_at": datetime.now().isoformat(),
        "summary": {
            "total": len(diagnostics),
            "web_search_triggered_count": web_count,
            "web_search_triggered_ratio": web_count / max(len(diagnostics), 1),
            "avg_elapsed_time_s": (
                sum(d["elapsed_time"] for d in diagnostics) / max(len(diagnostics), 1)
            ),
        },
        "per_query": diagnostics,
    }
    diagnostics_path.write_text(
        json.dumps(diag_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n  결과 저장: {result_path.name}")
    print(f"  진단 저장: {diagnostics_path.name}")
