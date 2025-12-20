import json
from typing import Dict, List, Tuple, Any
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from flow.state.chat_state import ChatState
from models.basic_chat.request import ChatRequest
from repositories.repository import repository


class RagChatGraph:
    def __init__(
            self,
            llm: ChatOpenAI,
            summary_prompt: PromptTemplate,
            base_chat_prompt: PromptTemplate,
            summary_token_threshold: int = 2500
    ):
        self._llm = llm
        self._summary_prompt = summary_prompt
        self._chat_prompt = base_chat_prompt
        self._summary_token_threshold = summary_token_threshold

        self._store = InMemoryStore
        self._checkpointer = InMemorySaver()

        self._search_cli = HybridRetriever()

    async def summarize_messages(
            self,
            messages: List[BaseMessage],
    )-> Tuple[List[BaseMessage], Dict[str, int]]:
        chat_history = "\n".join([f"{m.type}: {m.content}" for m in messages])
        summary_request = self._summary_prompt.format(chat_history=chat_history)

        summary = await self._llm.ainvoke(summary_request)
        usage = getattr(summary, "usage_metadata", None)

        normalized_usage: Dict[str, int] = {}

        if usage:
            normalized_usage = {
                "input_tokens": 0,
                "output_tokens": usage.get("output_tokens") or usage.get("total_tokens") or 0,
                "total_tokens": usage.get("output_tokens") or usage.get("total_tokens") or 0
            }
        return [AIMessage(content=f"[요약] {summary.content}")], normalized_usage or {}
    
    def build_graph(self):
        async def retrieve(state: ChatState) -> ChatState:
            query = state.user_latest_message()
            state.set_status("retrieving")

            retrieved_docs = self._search_cli.hybrid_search(query, top_k = 5)

            doc_context: Dict[str, Any] = {}

            async for doc in retrieved_docs:
                doc_context = {
                    "source": doc.metadata.get("source", "")
                    "page": doc.metadata.get("page", "")
                    "content": doc.metadata.get("content_kor", "")
                }

                state.add_documents(doc_context)

            state.set_status("retrieved")

            return state
        
        
        async def generate(state: ChatState) -> ChatState:
            state.set("generating")
            
            user_message = state.user_lagest_message()

            full_response = ""
            final_usage:Dict[str, int] = {}

            response_request = self._chat_prompt.format(
                chat_history = state.messages,
                message = user_message,
                context = state.document
            )

            async for token in self._llm.astream(response_request):
                                                 
                if token.content:
                    full_response += token.content

                if token.usage_metadata:
                    final_usage += token.usage_metadata

            state.add_ai_message(full_response)
            if token.usage_metadata:
                state.add_usage(final_usage)

            response_dict: Dict[str, Any] = {}

            try:
                response_dict = json.loads(full_response)
            except json.JSONDecodeError:
                response_dict = {"answer": full_response, "reference": []}

            state.update_response(response_dict)

            return state
        
        async def conditional_summarization(state: ChatState)->ChatState:
            total_tokens = state.total_tokens()

            should_summarize = (
                self._summary_token_threshold > 0
                and total_tokens >= self._summary_token_threshold
            )

            if should_summarize:
                state.set_status("summarizing")

                summary, token_usage = await self.summarize_messages(messages=state.messages)

                state.messages = summary

                if token_usage:
                    state.set_usage(token_usage)

                state.set_status("summarized")

            else:
                state.set_status("completed")

            return state
        
        async def emit_usage(state: ChatState) -> ChatState:

            return state
        
        graph = StateGraph(ChatState)
        graph.add_node("retrieve", retrieve)
        graph.add_node("generate", generate)
        graph.add_node("summarize", conditional_summarization)
        graph.add_node("emit_usage", emit_usage)

        graph.set_entry_point("retrieve")

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "summarize")
        graph.add_edge("summarize", "emit_usage")
        graph.add_edge("emit_usage", END)

        return graph.compile(checkpointer = self._checkpointer)
