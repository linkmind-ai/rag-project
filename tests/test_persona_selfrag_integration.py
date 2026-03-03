import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from graphs.rag_graph import RAGGraph
from models.state import Document, GraphState, RetrievedContext, SelfRagScores
from stores.memory_store import InMemoryStore


@pytest.fixture
def rag_graph() -> RAGGraph:
    graph = RAGGraph()
    graph._initialized = True
    graph._llm = object()
    return graph


@pytest.mark.asyncio
async def test_route_creative_query_is_minimal(rag_graph: RAGGraph) -> None:
    state = GraphState(query="소설 아이디어를 브레인스토밍 해줘", session_id="s1")
    result = await rag_graph._route_input_node(state)
    route = result["route"]

    assert route.task_type == "creative"
    assert route.retrieval_policy == "minimal"


@pytest.mark.asyncio
async def test_route_high_risk_query_forces_retrieval(rag_graph: RAGGraph) -> None:
    state = GraphState(query="의료 진단 관련 근거를 알려줘", session_id="s2")
    result = await rag_graph._route_input_node(state)
    route = result["route"]

    assert route.task_type == "factual"
    assert route.risk_level == "high"
    assert route.retrieval_policy == "forced"


@pytest.mark.asyncio
async def test_persona_bundle_builds_e0_and_m0(rag_graph: RAGGraph) -> None:
    docs = [
        Document(content="doc0 content", metadata={"title": "T0"}, doc_id="d0"),
        Document(content="doc1 content", metadata={"title": "T1"}, doc_id="d1"),
    ]

    async def fake_invoke(_prompt, data):
        if "session_summary" in data:
            return '{"rewritten_query":"rewritten q","source_plan":["wiki"]}'
        if "profile" in data and "documents" in data:
            return '{"ranked_indices":[1,0],"rerank_notes":["profile boost"]}'
        return "{}"

    async def fake_search(query: str, k: int, vector_weight: float):
        assert query == "rewritten q"
        return RetrievedContext(documents=docs, scores=[0.9, 0.8], total_hits=2)

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    with patch(
        "graphs.rag_graph.elasticsearch_store.hybrid_search", side_effect=fake_search
    ):
        state = GraphState(query="test query", session_id="s3")
        state.route.retrieval_policy = "forced"
        result = await rag_graph._build_persona_bundle_node(state)

    bundle = result["persona_bundle"]
    pool = result["global_message_pool"]

    assert bundle.top_docs[0].doc_id == "d1"
    assert pool.retrieval_plan["rewritten_query"] == "rewritten q"
    assert pool.rerank_notes == ["profile boost"]


@pytest.mark.asyncio
async def test_check_sufficiency_respects_loop_cap(rag_graph: RAGGraph) -> None:
    state = GraphState(query="q", session_id="s4")
    state.selfrag_scores = SelfRagScores(insufficiency_reasons=["low_utility"])
    state.loop_count = 0
    state.max_loops = 2

    r1 = await rag_graph._check_sufficiency_node(state)
    assert r1["next_action"] == "reinforce"

    state.loop_count = 2
    r2 = await rag_graph._check_sufficiency_node(state)
    assert r2["next_action"] == "finalize"


@pytest.mark.asyncio
async def test_identify_evidence_exception_fallback_returns_all_indices(
    rag_graph: RAGGraph,
) -> None:
    docs = [
        Document(content="a", metadata={}, doc_id="1"),
        Document(content="b", metadata={}, doc_id="2"),
    ]
    state = GraphState(query="q", session_id="s5", retrieved_docs=docs)

    with patch("asyncio.to_thread", side_effect=Exception("fail")):
        result = await rag_graph._identify_evidence_node(state)

    assert result["evidence_indices"] == [0, 1]


@pytest.mark.asyncio
async def test_feedback_updates_profile_explicitly() -> None:
    store = InMemoryStore()
    feedback = {
        "session_id": "s6",
        "rating": 2,
        "feedback_text": "더 근거가 필요해",
        "tags": ["factual", "citation"],
        "metadata": {"response_style": "evidence_first"},
    }

    await store.add_feedback_event("s6", feedback)
    profile = await store.update_profile_from_feedback("s6", feedback)

    assert profile["last_feedback_rating"] == 2
    assert "factual" in profile["preferred_topics"]
    assert profile["response_style"] == "evidence_first"
    assert profile["factuality_bias"] > 0.5
