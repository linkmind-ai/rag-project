from nodes import hyde_node, retrieval_node, response_node
from state import RAGState
from langgraph.graph import StateGraph


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("hyde", hyde_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("hyde")
    graph.add_edge("hyde", "retrieval")
    graph.add_edge("retrieval", "response")
    graph.set_finish_point("response")

    return graph.compile()