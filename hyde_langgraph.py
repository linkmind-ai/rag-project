from langgraph_utils.state import RAGState
from langgraph_utils.graph import build_graph


question = input("질문을 입력하세요: ")

initial_state: RAGState = {
    "question": question,
    "expanded_query": None,
    "contexts": None,
    "answer": None,
}

graph = build_graph()
final_state = graph.invoke(initial_state)

print(final_state["answer"])