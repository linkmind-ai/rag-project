from state import RAGState
from resources import client, config, hyde_prompt, final_prompt, searcher

def hyde_node(state: RAGState) -> RAGState:
    response = client.chat(
        model=config.generation_model_name,
        messages=[{
            "role": "user",
            "content": hyde_prompt.format(question=state["question"]),
        }]
    )
    state["expanded_query"] = response["message"]["content"]
    return state


def retrieval_node(state: RAGState) -> RAGState:
    contexts = searcher.search(state["expanded_query"], k=3)
    state["contexts"] = contexts
    return state


def response_node(state: RAGState) -> RAGState:
    response = client.chat(
        model=config.generation_model_name,
        messages=[{
            "role": "user",
            "content": final_prompt.format(
                question=state["question"],
                contexts=state["contexts"]
            ),
        }]
    )
    state["answer"] = response["message"]["content"]
    return state