import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from graphs.rag_graph import RAGGraph
from models.state import Document, GraphState, Message, RetrievedContext, SelfRagScores
from stores.memory_store import InMemoryStore


@pytest.fixture
def rag_graph() -> RAGGraph:
    graph = RAGGraph()
    graph._initialized = True
    graph._llm = object()
    return graph


@pytest.mark.asyncio
async def test_build_persona_bundle_limits_retrieval_query_to_search_relevant_context(
    rag_graph: RAGGraph,
) -> None:
    docs = [Document(content="doc0 content", metadata={"title": "T0"}, doc_id="d0")]

    async def fake_profile(_session_id: str) -> dict[str, object]:
        return {
            "preferred_topics": ["rag", "evaluation", "citations"],
            "explicit_notes": ["Explain with evidence"],
            "response_style": "evidence_first",
        }

    async def fake_invoke(_prompt: object, data: dict[str, object]) -> str:
        assert data["query"] == "tell me more"
        assert "Explain retrieval quality" in str(data["recent_user_context"])
        return '{"attach_context": true, "reason": "follow-up query"}'

    async def fake_search(query: str, k: int, vector_weight: float) -> RetrievedContext:
        assert query.startswith("tell me more")
        assert "Recent user context: Explain retrieval quality" in query
        assert "Topic hints: rag, evaluation" in query
        assert "response_style" not in query
        assert "evidence_first" not in query
        assert "Explain with evidence" not in query
        assert "assistant:Do not leak this reply" not in query
        assert k == 3
        assert vector_weight == 0.5
        return RetrievedContext(documents=docs, scores=[0.9], total_hits=1)

    state = GraphState(
        query="original query",
        session_id="s1",
        next_query="tell me more",
        chat_history=[
            Message(role="user", content="Explain retrieval quality"),
            Message(role="assistant", content="Do not leak this reply"),
            Message(role="user", content="Focus on search metrics"),
        ],
    )

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    with patch(
        "graphs.rag_graph.memory_store.get_user_profile",
        side_effect=fake_profile,
    ), patch(
        "graphs.rag_graph.elasticsearch_store.hybrid_search",
        side_effect=fake_search,
    ):
        result = await rag_graph._build_persona_bundle_node(state)

    assert result["retrieved_docs"][0].doc_id == "d0"
    assert result["retrieval_scores"] == [0.9]
    assert result["last_retrieval_query"] == result["retrieval_query"]
    assert "assistant:Do not leak this reply" in result["session_summary"]
    assert "Response style: evidence_first" in result["generation_hints"]
    assert "Preferred topics: rag, evaluation, citations" in result["generation_hints"]
    assert "Notes: Explain with evidence" in result["generation_hints"]


@pytest.mark.asyncio
async def test_build_persona_bundle_keeps_specific_query_clean(
    rag_graph: RAGGraph,
) -> None:
    docs = [Document(content="doc0 content", metadata={"title": "T0"}, doc_id="d0")]
    query = "How should we evaluate retrieval precision for a RAG system?"

    async def fake_profile(_session_id: str) -> dict[str, object]:
        return {
            "preferred_topics": ["rag", "evaluation"],
            "explicit_notes": ["Keep answers concise"],
            "response_style": "balanced",
        }

    async def fake_invoke(_prompt: object, data: dict[str, object]) -> str:
        assert data["query"] == query
        return '{"attach_context": false, "reason": "standalone query"}'

    async def fake_search(query: str, k: int, vector_weight: float) -> RetrievedContext:
        assert query == "How should we evaluate retrieval precision for a RAG system?"
        return RetrievedContext(documents=docs, scores=[0.7], total_hits=1)

    state = GraphState(
        query=query,
        session_id="s2",
        chat_history=[
            Message(role="user", content="Earlier question"),
            Message(role="assistant", content="Earlier answer"),
        ],
    )

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    with patch(
        "graphs.rag_graph.memory_store.get_user_profile",
        side_effect=fake_profile,
    ), patch(
        "graphs.rag_graph.elasticsearch_store.hybrid_search",
        side_effect=fake_search,
    ):
        result = await rag_graph._build_persona_bundle_node(state)

    assert result["retrieval_query"] == state.query


@pytest.mark.asyncio
async def test_build_persona_bundle_skips_attachment_when_llm_response_is_invalid(
    rag_graph: RAGGraph,
) -> None:
    docs = [Document(content="doc0 content", metadata={"title": "T0"}, doc_id="d0")]

    async def fake_profile(_session_id: str) -> dict[str, object]:
        return {
            "preferred_topics": ["rag", "evaluation"],
            "explicit_notes": ["Keep answers concise"],
            "response_style": "balanced",
        }

    async def fake_invoke(_prompt: object, _data: dict[str, object]) -> str:
        return "not-json"

    async def fake_search(query: str, k: int, vector_weight: float) -> RetrievedContext:
        assert query == "Can you expand this?"
        return RetrievedContext(documents=docs, scores=[0.7], total_hits=1)

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    state = GraphState(
        query="Can you expand this?",
        session_id="s-extra",
        chat_history=[Message(role="user", content="We were discussing retrieval metrics")],
    )

    with patch(
        "graphs.rag_graph.memory_store.get_user_profile",
        side_effect=fake_profile,
    ), patch(
        "graphs.rag_graph.elasticsearch_store.hybrid_search",
        side_effect=fake_search,
    ):
        result = await rag_graph._build_persona_bundle_node(state)

    assert result["retrieval_query"] == state.query


@pytest.mark.asyncio
async def test_selfrag_critique_uses_generation_hints_and_normalizes_same_query(
    rag_graph: RAGGraph,
) -> None:
    docs = [Document(content="supported answer evidence", metadata={}, doc_id="1")]
    seen_payload: dict[str, object] = {}

    async def fake_invoke(_prompt: object, data: dict[str, object]) -> str:
        seen_payload.update(data)
        return """
        {
          "answer": "Grounded answer.",
          "is_sufficient": false,
          "utility_score": 4.5,
          "confidence": 0.82,
          "insufficiency_reasons": ["missing_date"],
          "next_query": "Original query!"
        }
        """

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    state = GraphState(
        query="Original query",
        session_id="s3",
        retrieved_docs=docs,
        retrieval_scores=[0.8],
        retrieval_query="Original query\nRecent user context: prior topic",
        session_summary="user:Original query",
        generation_hints=(
            "User preferences:\n"
            "- Response style: evidence_first\n"
            "- Notes: Explain with citations"
        ),
    )

    result = await rag_graph._self_critique_node(state)
    scores = result["selfrag_scores"]

    assert seen_payload["generation_hints"] == state.generation_hints
    assert "user_profile" not in seen_payload
    assert result["answer"] == "Grounded answer."
    assert result["is_sufficient"] is False
    assert result["next_query"] == "Original query"
    assert scores.utility_score == 4.5
    assert scores.confidence == 0.82


@pytest.mark.asyncio
async def test_selfrag_critique_falls_back_when_json_is_invalid(
    rag_graph: RAGGraph,
) -> None:
    async def fake_invoke(_prompt: object, _data: dict[str, object]) -> str:
        return "not-json"

    rag_graph._invoke = fake_invoke  # type: ignore[assignment]

    state = GraphState(
        query="Question",
        session_id="s4",
        retrieved_docs=[],
        retrieval_scores=[],
    )

    result = await rag_graph._self_critique_node(state)

    assert result["is_sufficient"] is False
    assert result["next_query"] == "Question"
    assert "No relevant documents were retrieved" in result["answer"]


@pytest.mark.asyncio
async def test_check_sufficiency_respects_loop_cap(rag_graph: RAGGraph) -> None:
    state = GraphState(
        query="q",
        session_id="s5",
        selfrag_scores=SelfRagScores(
            utility_score=2.0,
            confidence=0.3,
            insufficiency_reasons=["low_utility"],
            next_query="better q",
        ),
        next_query="better q",
        is_sufficient=False,
        loop_count=0,
        max_loops=1,
    )

    retry_result = await rag_graph._check_sufficiency_node(state)
    assert retry_result["next_action"] == "retry"
    assert retry_result["loop_count"] == 1
    assert retry_result["next_query"] == "better q"

    state.loop_count = 1
    finalize_result = await rag_graph._check_sufficiency_node(state)
    assert finalize_result["next_action"] == "finalize"


@pytest.mark.asyncio
async def test_feedback_updates_profile_explicitly() -> None:
    store = InMemoryStore()
    feedback = {
        "session_id": "s6",
        "rating": 2,
        "feedback_text": "Please cite evidence.",
        "tags": ["factual", "citation"],
        "metadata": {"response_style": "evidence_first"},
    }

    await store.add_feedback_event("s6", feedback)
    profile = await store.update_profile_from_feedback("s6", feedback)

    assert profile["last_feedback_rating"] == 2
    assert "factual" in profile["preferred_topics"]
    assert profile["response_style"] == "evidence_first"
    assert profile["factuality_bias"] > 0.5
