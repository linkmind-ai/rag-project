import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from models.state import Document
from routers import query as query_router


def test_query_endpoint_keeps_response_shape(monkeypatch) -> None:
    async def fake_process_query(session_id: str, query: str, use_history: bool = True):
        assert session_id == "session-1"
        assert query == "질문"
        assert use_history is False
        return {
            "answer": "답변",
            "retrieved_docs": [
                Document(content="문서1", metadata={"title": "A"}, doc_id="d1"),
                Document(content="문서2", metadata={"title": "B"}, doc_id="d2"),
            ],
            "meta": {
                "loop_count": 1,
                "is_sufficient": False,
                "last_retrieval_query": "질문\n\nRecent conversation: 없음",
                "retrieval_scores": [0.9, 0.8],
                "selfrag_scores": {
                    "utility_score": 3.0,
                    "confidence": 0.5,
                    "insufficiency_reasons": ["missing_detail"],
                    "next_query": "더 구체적인 질문",
                },
            },
        }

    monkeypatch.setattr(query_router.rag_service, "process_query", fake_process_query)

    app = FastAPI()
    app.include_router(query_router.router)
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"session_id": "session-1", "query": "질문", "use_history": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "답변"
    assert len(payload["sources"]) == 2
    assert payload["sources"][0]["index"] == 0
    assert payload["meta"]["loop_count"] == 1
    assert payload["meta"]["selfrag_scores"]["next_query"] == "더 구체적인 질문"


def test_query_stream_emits_new_event_set(monkeypatch) -> None:
    async def fake_process_query_stream(
        session_id: str, query: str, use_history: bool = True
    ):
        assert session_id == "session-2"
        assert query == "스트림 질문"
        assert use_history is True
        yield {"type": "retrieve_start", "message": "Document retrieval started"}
        yield {"type": "retrieve_end", "doc_count": 2}
        yield {
            "type": "self_critique_start",
            "message": "Generating answer and checking sufficiency",
        }
        yield {"type": "self_critique_end", "is_sufficient": False}
        yield {"type": "retry", "loop_count": 1, "next_query": "보강 질문"}
        yield {"type": "done", "full_response": "최종 답변", "meta": {"loop_count": 1}}

    monkeypatch.setattr(
        query_router.rag_service, "process_query_stream", fake_process_query_stream
    )

    app = FastAPI()
    app.include_router(query_router.router)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/query/stream",
        json={"session_id": "session-2", "query": "스트림 질문", "use_history": True},
    ) as response:
        body = "".join(chunk for chunk in response.iter_text())

    assert response.status_code == 200
    assert '"type": "retrieve_start"' in body
    assert '"type": "self_critique_start"' in body
    assert '"type": "retry"' in body
    assert '"type": "done"' in body
    assert "finalize" not in body
    assert "evidence_end" not in body
