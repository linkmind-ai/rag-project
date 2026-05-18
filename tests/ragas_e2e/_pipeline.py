"""
_pipeline.py — 전체 RAG 파이프라인 실행 로직

RAGService.process_query()를 통해 LangGraph 전체 7노드를 실행하고
RAGAS EvaluationDataset과 파이프라인 진단 목록을 반환.

노드 실행 순서:
  N1: hyde            — 가상 문서 생성 (rewriter LLM)
  N2: retrieve        — HyDE+query 결합 hybrid_search
  N3: grade_documents — 쿼리-문서 관련성 grading (grader LLM)
  N4: query_rewrite   — 웹 검색용 쿼리 재작성 (web_search=True 시)
  N5: search_web      — Tavily 웹 검색 (web_search=True 시)
  N6: generate        — 운영 chat_prompt.py 사용 답변 생성 (main LLM)
  N7: identify_evidence — 하이브리드 근거 식별

[PyCharm 디버깅]
  _run_single_sample() 내부 '# --- BREAKPOINT ---' 위치에서
  Variables 패널로 각 노드 출력 확인 가능.
  Step Into(F7) → RAGService → rag_graph → 각 노드 함수 추적 가능.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

# apps/ 를 Python path에 추가 (conftest.py가 없는 standalone 실행 대응)
_APPS_DIR = Path(__file__).parent.parent.parent / "apps"
if str(_APPS_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_DIR))

# process_query() 단일 쿼리 최대 대기 시간(초)
_QUERY_TIMEOUT_S: int = 120


# ── 노드 출력 추출 ────────────────────────────────────────────────────────────


def _extract_node_outputs(raw_result: dict[str, Any]) -> dict[str, Any]:
    """
    raw_result에서 각 LangGraph 노드 출력을 명시적 변수로 분리.

    Returns:
        노드별 출력이 담긴 dict.
        filtered_docs=0(grade 후 관련 문서 없음)이면 contexts=[]이며,
        이때 answer에는 LLM의 "정보를 찾을 수 없습니다" 거절 답변이 담긴다.
    """
    answer: str = raw_result["answer"]  # N6: generate
    hypothetical_doc: str = raw_result.get("hypothetical_doc", "")  # N1: hyde
    web_search_triggered: bool = raw_result.get("web_search", False)  # N3: grade
    all_docs: list[Any] = raw_result.get("all_docs", [])  # N2: retrieve
    evidence_indices: list[int] = raw_result.get("evidence_indices", [])  # N7
    evidence_docs: list[Any] = raw_result.get("evidence_docs", [])
    elapsed_time: float = raw_result.get("elapsed_time", 0.0)

    # --- BREAKPOINT: all_docs로 검색 문서 내용 확인 ---
    contexts: list[str] = [
        doc.content if hasattr(doc, "content") else doc["content"] for doc in all_docs
    ]

    # contexts가 비어도 스킵하지 않는다 — filtered_docs=0일 때 LLM은
    # "Notion 페이지에서 해당 정보를 찾을 수 없습니다."를 답변하며,
    # 이 거절 답변도 RAGAS 평가 대상(올바른 거절 여부)에 포함되어야 한다.

    return {
        "answer": answer,
        "hypothetical_doc": hypothetical_doc,
        "web_search_triggered": web_search_triggered,
        "evidence_indices": evidence_indices,
        "evidence_doc_count": len(evidence_docs),
        "elapsed_time": elapsed_time,
        "contexts": contexts,
    }


def _print_node_summary(query: str, node_outputs: dict[str, Any]) -> None:
    """노드별 진단 정보를 콘솔에 출력."""
    hypothetical_doc = node_outputs["hypothetical_doc"]
    web_search_triggered = node_outputs["web_search_triggered"]
    contexts = node_outputs["contexts"]
    evidence_indices = node_outputs["evidence_indices"]
    elapsed_time = node_outputs["elapsed_time"]
    answer = node_outputs["answer"]

    print(f"\n  Q : {query[:65]}...")
    print(
        f"  N1 HyDE     : {hypothetical_doc[:60]}..."
        if hypothetical_doc
        else "  N1 HyDE     : (없음)"
    )
    print(
        f"  N3 Grade    : {'웹검색 경로 (N4→N5)' if web_search_triggered else '생성 경로 (N6)'}"
    )
    print(
        f"  N2 Retrieve : {len(contexts)}개 | N7 Evidence: {evidence_indices} | ⏱ {elapsed_time:.1f}s"
    )
    print(f"  N6 Answer   : {answer[:80]}...")


# ── 단일 샘플 실행 ────────────────────────────────────────────────────────────


async def _run_single_sample(
    item: dict[str, Any],
    session_id: str,
    service: Any,
) -> dict[str, Any] | None:
    """
    golden_set 항목 1개를 RAGService.process_query()로 실행.

    Args:
        item: golden_set 항목 (query, reference, reference_contexts 포함)
        session_id: 고유 세션 ID
        service: 초기화된 RAGService 인스턴스 (루프 외부에서 1회 생성)

    Returns:
        성공 시: RAGAS 필드 + 파이프라인 진단 필드가 담긴 dict
        실패 시: None
    """
    query: str = item["query"]

    try:
        # ── 전체 파이프라인 실행 (타임아웃: _QUERY_TIMEOUT_S) ──────────────
        # Step Into(F7) → RAGService.process_query → rag_graph.ainvoke
        # --- BREAKPOINT: raw_result로 전체 파이프라인 출력 확인 ---
        raw_result = await asyncio.wait_for(
            service.process_query(
                session_id=session_id,
                query=query,
                use_history=False,  # RAGAS 평가: 단일 쿼리, 멀티턴 이력 없음
            ),
            timeout=_QUERY_TIMEOUT_S,
        )

        node_outputs = _extract_node_outputs(raw_result)

    except asyncio.TimeoutError:
        print(f"  ⏱ 타임아웃 스킵 ({_QUERY_TIMEOUT_S}s 초과): {query[:50]!r}")
        return None
    except Exception as exc:
        print(f"  ❌ 파이프라인 오류 (query: {query[:50]!r}): {exc}")
        return None

    # --- BREAKPOINT: node_outputs으로 모든 노드 출력 확인 ---
    _print_node_summary(query, node_outputs)

    return {
        # RAGAS SingleTurnSample 구성용
        "query": query,
        "answer": node_outputs["answer"],
        "contexts": node_outputs["contexts"],
        "reference": item.get("reference") or item.get("summary", ""),
        "reference_contexts": item.get("reference_contexts", []),
        # 파이프라인 진단용 (ragas_e2e_diagnostics.json 저장)
        "hypothetical_doc": node_outputs["hypothetical_doc"],
        "web_search_triggered": node_outputs["web_search_triggered"],
        "evidence_indices": node_outputs["evidence_indices"],
        "evidence_doc_count": node_outputs["evidence_doc_count"],
        "elapsed_time": node_outputs["elapsed_time"],
    }


# ── 전체 golden_set 실행 ──────────────────────────────────────────────────────


async def build_e2e_dataset(
    golden_set: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """
    golden_set 전체를 순차 실행하여 RAGAS EvaluationDataset과 진단 목록 반환.

    RAGService를 1회만 초기화하여 루프 전체에서 재사용 (성능 최적화).

    반환값:
        (EvaluationDataset, diagnostics)
        - EvaluationDataset : ragas.evaluate()에 전달
        - diagnostics       : 노드별 중간 결과, JSON 저장용

    PyCharm 디버깅:
        _run_single_sample()에 Step Into(F7) → 각 LangGraph 노드 함수 추적
    """
    from ragas import EvaluationDataset, SingleTurnSample
    from services.service import RAGService

    # RAGService 1회 초기화 — 루프 내 재생성 방지 (V5 교정)
    service = RAGService()
    await service.initialize()

    samples: list[SingleTurnSample] = []
    diagnostics: list[dict[str, Any]] = []  # --- BREAKPOINT: 전체 진단 확인 ---

    for i, item in enumerate(golden_set):
        session_id = f"ragas-e2e-{uuid.uuid4()}"
        print(f"\n[{i + 1}/{len(golden_set)}] 파이프라인 실행...")

        # --- BREAKPOINT: item으로 query/reference/reference_contexts 확인 ---
        result = await _run_single_sample(item, session_id, service)

        if result is None:
            continue

        # --- BREAKPOINT: result로 모든 노드 출력 확인 ---
        sample_kwargs: dict[str, Any] = {
            "user_input": result["query"],
            "retrieved_contexts": result["contexts"],
            "response": result["answer"],
            "reference": result["reference"],
        }
        if result["reference_contexts"]:
            sample_kwargs["reference_contexts"] = result["reference_contexts"]

        samples.append(SingleTurnSample(**sample_kwargs))
        diagnostics.append(
            {
                "index": i,
                "session_id": session_id,
                "query": result["query"],
                "hypothetical_doc": result["hypothetical_doc"],
                "web_search_triggered": result["web_search_triggered"],
                "evidence_indices": result["evidence_indices"],
                "evidence_doc_count": result["evidence_doc_count"],
                "context_count": len(result["contexts"]),
                "elapsed_time": result["elapsed_time"],
            }
        )

    # ── 실행 요약 ─────────────────────────────────────────────────────────
    # --- BREAKPOINT: diagnostics 전체 통계 확인 ---
    web_count = sum(1 for d in diagnostics if d["web_search_triggered"])
    avg_elapsed = sum(d["elapsed_time"] for d in diagnostics) / max(len(diagnostics), 1)

    print(f"\n\n{'=' * 62}")
    print("E2E 파이프라인 실행 요약")
    print(f"  총 샘플       : {len(samples)} / {len(golden_set)}")
    print(
        f"  웹검색 실행   : {web_count}건 ({web_count / max(len(samples), 1) * 100:.1f}%)"
    )
    print(f"  평균 처리시간 : {avg_elapsed:.1f}s / 쿼리")
    print(f"{'=' * 62}\n")

    assert samples, "샘플 생성 실패 — ES/Ollama 연결 및 TAVILY_API_KEY 확인"
    return EvaluationDataset(samples=samples), diagnostics
