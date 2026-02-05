"""
검색 품질 평가 스크립트

Golden Set을 기반으로 hybrid_search의 검색 품질을 측정합니다.
메트릭: Hit@K, MRR (Mean Reciprocal Rank), Precision@K
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# apps 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from stores.vector_store import elasticsearch_store
from common.config import settings


@dataclass
class SearchResult:
    """단일 쿼리 검색 결과"""

    query: str
    expected_page_id: str
    keywords: List[str]
    summary: str
    retrieved_page_ids: List[str]
    scores: List[float]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float


@dataclass
class EvaluationReport:
    """전체 평가 리포트"""

    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    results: List[SearchResult]


def load_golden_set(path: str) -> List[Dict[str, Any]]:
    """Golden Set JSON 파일 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_reciprocal_rank(
    expected_id: str, retrieved_ids: List[str]
) -> float:
    """Reciprocal Rank 계산 (1/rank, 없으면 0)"""
    for idx, page_id in enumerate(retrieved_ids):
        if expected_id in page_id:
            return 1.0 / (idx + 1)
    return 0.0


def check_hit(expected_id: str, retrieved_ids: List[str], k: int) -> bool:
    """Hit@K 체크 (상위 K개 내에 정답 존재 여부)"""
    for page_id in retrieved_ids[:k]:
        if expected_id in page_id:
            return True
    return False


async def evaluate_single_query(
    query_data: Dict[str, Any], k: int = 5
) -> SearchResult:
    """단일 쿼리 평가"""
    query = query_data["query"]
    expected_page_id = query_data["expected_page_id"]
    keywords = query_data.get("keywords", [])
    summary = query_data.get("summary", "")

    # hybrid_search 실행
    context = await elasticsearch_store.hybrid_search(
        query=query, k=k, vector_weight=0.5
    )

    # 검색된 문서에서 page_id 추출
    retrieved_page_ids = []
    for doc in context.documents:
        page_id = doc.metadata.get("page_id", "")
        origin_doc_id = doc.metadata.get("origin_doc_id", "")
        retrieved_page_ids.append(page_id or origin_doc_id or doc.doc_id)

    # 메트릭 계산
    hit_at_1 = check_hit(expected_page_id, retrieved_page_ids, 1)
    hit_at_3 = check_hit(expected_page_id, retrieved_page_ids, 3)
    hit_at_5 = check_hit(expected_page_id, retrieved_page_ids, 5)
    reciprocal_rank = calculate_reciprocal_rank(expected_page_id, retrieved_page_ids)

    return SearchResult(
        query=query,
        expected_page_id=expected_page_id,
        keywords=keywords,
        summary=summary,
        retrieved_page_ids=retrieved_page_ids,
        scores=context.scores,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        reciprocal_rank=reciprocal_rank,
    )


async def evaluate_all(golden_set: List[Dict[str, Any]], k: int = 5) -> EvaluationReport:
    """전체 Golden Set 평가"""
    await elasticsearch_store.initialize()

    results: List[SearchResult] = []
    for query_data in golden_set:
        result = await evaluate_single_query(query_data, k)
        results.append(result)

    # 집계 메트릭 계산
    total = len(results)
    hit_at_1 = sum(1 for r in results if r.hit_at_1) / total if total > 0 else 0
    hit_at_3 = sum(1 for r in results if r.hit_at_3) / total if total > 0 else 0
    hit_at_5 = sum(1 for r in results if r.hit_at_5) / total if total > 0 else 0
    mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0

    return EvaluationReport(
        total_queries=total,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        mrr=mrr,
        results=results,
    )


def print_report(report: EvaluationReport, verbose: bool = False) -> None:
    """평가 리포트 출력"""
    print("\n" + "=" * 60)
    print("검색 품질 평가 리포트")
    print("=" * 60)

    print(f"\n총 쿼리 수: {report.total_queries}")
    print(f"\nHit@1: {report.hit_at_1 * 100:.1f}%")
    print(f"Hit@3: {report.hit_at_3 * 100:.1f}%")
    print(f"Hit@5: {report.hit_at_5 * 100:.1f}%")
    print(f"MRR:   {report.mrr:.3f}")

    print("\n" + "-" * 60)
    print("개별 쿼리 결과")
    print("-" * 60)

    for idx, result in enumerate(report.results, 1):
        status = "O" if result.hit_at_3 else "X"
        print(f"\n[{idx}] [{status}] {result.summary}")
        print(f"    Query: {result.query[:50]}...")
        print(f"    Keywords: {', '.join(result.keywords)}")
        print(f"    RR: {result.reciprocal_rank:.2f}")

        if verbose:
            print(f"    Retrieved IDs: {result.retrieved_page_ids}")
            print(f"    Scores: {[f'{s:.3f}' for s in result.scores]}")

    # 실패 케이스 요약
    failed = [r for r in report.results if not r.hit_at_3]
    if failed:
        print("\n" + "-" * 60)
        print(f"실패 케이스 ({len(failed)}건)")
        print("-" * 60)
        for result in failed:
            print(f"  - {result.summary}: {result.keywords}")


async def main() -> None:
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="검색 품질 평가")
    parser.add_argument(
        "--golden-set",
        type=str,
        default=str(Path(__file__).parent / "golden_set.json"),
        help="Golden Set JSON 파일 경로",
    )
    parser.add_argument(
        "--k", type=int, default=5, help="검색 결과 개수 (기본값: 5)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="상세 출력"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="결과 JSON 저장 경로"
    )

    args = parser.parse_args()

    # Golden Set 로드
    golden_set = load_golden_set(args.golden_set)
    print(f"Golden Set 로드 완료: {len(golden_set)}개 쿼리")

    # 평가 실행
    report = await evaluate_all(golden_set, k=args.k)

    # 리포트 출력
    print_report(report, verbose=args.verbose)

    # JSON 저장
    if args.output:
        output_data = {
            "total_queries": report.total_queries,
            "hit_at_1": report.hit_at_1,
            "hit_at_3": report.hit_at_3,
            "hit_at_5": report.hit_at_5,
            "mrr": report.mrr,
            "results": [
                {
                    "query": r.query,
                    "summary": r.summary,
                    "keywords": r.keywords,
                    "expected_page_id": r.expected_page_id,
                    "retrieved_page_ids": r.retrieved_page_ids,
                    "scores": r.scores,
                    "hit_at_1": r.hit_at_1,
                    "hit_at_3": r.hit_at_3,
                    "hit_at_5": r.hit_at_5,
                    "reciprocal_rank": r.reciprocal_rank,
                }
                for r in report.results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.output}")

    # 연결 종료
    await elasticsearch_store.close()


if __name__ == "__main__":
    asyncio.run(main())
