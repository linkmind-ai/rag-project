from typing import Dict, List, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from Langgraph.store.memory import InMemoryStore

from src.apps.flow.state.chat_state import ChatState


class BasicChatGraph:
    def __init__(
            self,
            llm: ChatOpenAI,
            summary_prompt: PromptTemplate,
            summary_token_threshold: int = 2500
    ):
        self._llm = llm
        self._summary_prompt = summary_prompt
        self._summary_token_threshold = summary_token_threshold

        self._store = InMemoryStore
        self._checkpointer = InMemorySaver()

    async def summarize_message(
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
        async def generate(state:ChatState) -> ChatState:

            full_response = ""
            final_usage:Dict[str, int] = {}

            async for token in self._llm.astream(state.messages):
                if token.content:
                    full_response += token.content

                if token.usage_metadata:
                    final_usage += token.usage_metadata

            state.add_ai_message(full_response)
            if token.usage_metadata:
                state.add_usage(final_usage)

            return state
        
        async def conditional_summarization(state: ChatState)->ChatState:
            total_tokens = state.total_tokens()

            should_summarize = (
                self._summary_token_threshold
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
        graph.add_node("generate", generate)
        graph.add_node("summarize", conditional_summarization)
        graph.add_node("emit_usage", emit_usage)
        graph.set_entry_point("generate")
        graph.add_edge("generate", "summarize")
        graph.add_edge("summarize", "emit_usage")
        graph.add_edge("emit_usage", END)

        return graph.compile(checkpointer = self._checkpointer)
