import sys
from pathlib import Path
import pytest
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graphs.rag_graph import RAGGraph
from models.state import Document, Message


@pytest.fixture
def graph():
    """CRAGGraph 인스턴스 생성"""
    return RAGGraph()


# --------------------------------------------------
# keyword extraction
# --------------------------------------------------


def test_extract_keywords_basic(graph):
    text = "AI 기술은 빠르게 발전하고 있다"
    keywords = graph._extract_keywords(text)

    assert isinstance(keywords, set)
    assert "ai" in keywords
    assert "빠르게" in keywords


# --------------------------------------------------
# keyword overlap
# --------------------------------------------------


def test_calculate_keyword_overlap(graph):

    answer = "AI 기술 발전"
    doc = "AI 기술이 빠르게 발전하고 있다"

    ratio, matched = graph._calculate_keyword_overlap(answer, doc)

    assert ratio > 0
    assert "ai" in matched
    assert isinstance(matched, set)


# --------------------------------------------------
# keyword based evidence
# --------------------------------------------------


def test_get_keyword_based_evidence(graph):

    answer = "AI 기술 발전"

    docs = [
        Document(doc_id="1", content="AI 기술이 빠르게 발전하고 있다", metadata={}),
        Document(doc_id="2", content="요리 레시피 소개", metadata={}),
    ]

    candidates = graph._get_keyword_based_evidence(answer, docs)

    assert len(candidates) == 1
    assert candidates[0][0] == 0


# --------------------------------------------------
# context build
# --------------------------------------------------


def test_build_context(graph):

    docs = [
        Document(doc_id="1", content="Document A", metadata={}),
        Document(doc_id="2", content="Document B", metadata={}),
    ]

    context = graph._build_context(docs)

    assert "Document 1" in context
    assert "Document 2" in context
    assert "Document A" in context


# --------------------------------------------------
# chat history conversion
# --------------------------------------------------


def test_prepare_messages(graph):

    history = [
        Message(role="user", content="안녕"),
        Message(role="assistant", content="안녕하세요"),
    ]

    messages = graph._prepare_messages(history)

    assert len(messages) == 2
    assert messages[0].content == "안녕"
    assert messages[1].content == "안녕하세요"


# --------------------------------------------------
# decision routing
# --------------------------------------------------


def test_decide_to_web_search(graph):

    state = SimpleNamespace(web_search=True)

    decision = graph._decide_to_web_search(state)

    assert decision == "query_rewrite"


def test_decide_to_generate(graph):

    state = SimpleNamespace(web_search=False)

    decision = graph._decide_to_web_search(state)

    assert decision == "prompt_compression"


# --------------------------------------------------
# evidence parsing
# --------------------------------------------------


@pytest.mark.asyncio
async def test_parse_evidence_response_json(graph):

    response = """
    ```json
    {"evidence_indices": [0,2]}
    ```
    """

    indices = await graph._parse_evidence_response(response)

    assert indices == [0, 2]


@pytest.mark.asyncio
async def test_parse_evidence_response_list(graph):

    response = "[1,3]"

    indices = await graph._parse_evidence_response(response)

    assert indices == [1, 3]


@pytest.mark.asyncio
async def test_parse_evidence_response_fallback(graph):

    response = "evidence indices are 2 and 4"

    indices = await graph._parse_evidence_response(response)

    assert indices == [2, 4]


# --------------------------------------------------
# retrieve node (mock vector store)
# --------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node(monkeypatch, graph):

    class DummyStore:
        async def hybrid_search(self, query, k, vector_weight):

            class Result:
                documents = [
                    Document(doc_id="test_doc", content="AI 관련 문서", metadata={})
                ]

            return Result()

    monkeypatch.setattr("graphs.rag_graph.elasticsearch_store", DummyStore())

    state = SimpleNamespace(query="AI")

    result = await graph._retrieve_node(state)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 1
