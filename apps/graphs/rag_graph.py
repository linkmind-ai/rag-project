from __future__ import annotations

import asyncio
import json
import re
from types import TracebackType
from typing import Any

from common.config import settings
from langchain_community.llms import Ollama
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from models.state import Document, GraphState, Message, SelfRagScores
from prompts.context_attachment_prompt import _CONTEXT_ATTACHMENT_PROMPT
from prompts.selfrag_critique_prompt import _SELFRAG_CRITIQUE_PROMPT
from stores.memory_store import memory_store
from stores.vector_store import elasticsearch_store


class RAGGraph:
    def __init__(self) -> None:
        self._llm: Ollama | None = None
        self._graph: CompiledStateGraph | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.context_attachment_prompt = _CONTEXT_ATTACHMENT_PROMPT
        self.selfrag_critique_prompt = _SELFRAG_CRITIQUE_PROMPT

    async def __aenter__(self) -> "RAGGraph":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                headers=settings.get_ollama_headers(),
                temperature=0.4,
            )
            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        workflow = StateGraph(GraphState)
        workflow.add_node("build_persona_bundle", self._build_persona_bundle_node)
        workflow.add_node("self_critique", self._self_critique_node)
        workflow.add_node("check_sufficiency", self._check_sufficiency_node)
        workflow.set_entry_point("build_persona_bundle")
        workflow.add_edge("build_persona_bundle", "self_critique")
        workflow.add_edge("self_critique", "check_sufficiency")
        workflow.add_conditional_edges(
            "check_sufficiency",
            self._route_after_sufficiency,
            {"retry": "build_persona_bundle", "finalize": END},
        )
        self._graph = workflow.compile()

    def _route_after_sufficiency(self, state: GraphState) -> str:
        return state.next_action

    async def _invoke(self, prompt: Any, data: dict[str, Any]) -> str:
        return str(await asyncio.to_thread((prompt | self._llm).invoke, data))

    def _parse_json(self, text: str) -> dict[str, Any]:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        candidate = match.group(1) if match else text
        if not match:
            obj = re.search(r"\{.*\}", text, re.DOTALL)
            if obj:
                candidate = obj.group(0)
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def _truncate(self, text: str, limit: int = 120) -> str:
        collapsed = " ".join(text.split())
        return collapsed[:limit]

    def _build_session_summary(self, chat_history: list[Message]) -> str:
        recent_messages = chat_history[-6:]
        return " | ".join(
            f"{message.role}:{self._truncate(message.content, 100)}"
            for message in recent_messages
        )

    def _build_recent_user_context(self, chat_history: list[Message]) -> str:
        user_messages = [
            self._truncate(message.content, 120)
            for message in chat_history
            if message.role == "user" and message.content.strip()
        ]
        return " | ".join(user_messages[-2:])

    async def _should_attach_context_with_llm(
        self,
        query: str,
        recent_user_context: str,
        session_summary: str,
        preferred_topics: list[Any],
    ) -> bool:
        if not query.strip() or not recent_user_context.strip():
            return False

        try:
            parsed = self._parse_json(
                await self._invoke(
                    self.context_attachment_prompt,
                    {
                        "query": query,
                        "recent_user_context": recent_user_context,
                        "session_summary": session_summary or "No recent conversation.",
                        "preferred_topics": ", ".join(
                            str(topic) for topic in preferred_topics[:2]
                        )
                        or "None",
                    },
                )
            )
        except Exception:
            parsed = {}

        return bool(parsed.get("attach_context"))

    async def _build_retrieval_query(
        self,
        base_query: str,
        chat_history: list[Message],
        session_summary: str,
        profile: dict[str, Any],
    ) -> str:
        trimmed_query = base_query.strip() or base_query
        parts = [trimmed_query]

        recent_user_context = self._build_recent_user_context(chat_history)
        preferred_topics = profile.get("preferred_topics") or []
        attach_context = await self._should_attach_context_with_llm(
            query=trimmed_query,
            recent_user_context=recent_user_context,
            session_summary=session_summary,
            preferred_topics=preferred_topics,
        )

        if attach_context:
            parts.append(f"Recent user context: {recent_user_context}")

            if preferred_topics:
                topic_hints = ", ".join(str(topic) for topic in preferred_topics[:2])
                if topic_hints:
                    parts.append(f"Topic hints: {topic_hints}")

        return "\n".join(part for part in parts if part)

    def _build_generation_hints(self, profile: dict[str, Any]) -> str:
        lines: list[str] = []

        response_style = profile.get("response_style")
        if isinstance(response_style, str) and response_style.strip():
            lines.append(f"- Response style: {response_style.strip()}")

        preferred_topics = profile.get("preferred_topics") or []
        if preferred_topics:
            topic_list = ", ".join(str(topic) for topic in preferred_topics[:3])
            lines.append(f"- Preferred topics: {topic_list}")

        explicit_notes = profile.get("explicit_notes") or []
        if explicit_notes:
            notes = " | ".join(
                self._truncate(str(note), 80) for note in explicit_notes[-2:]
            )
            lines.append(f"- Notes: {notes}")

        if not lines:
            return ""
        return "User preferences:\n" + "\n".join(lines)

    def _serialize_docs(
        self, docs: list[Document], scores: list[float], limit: int = 500
    ) -> str:
        serialized = []
        for index, doc in enumerate(docs):
            score = scores[index] if index < len(scores) else 0.0
            serialized.append(
                "\n".join(
                    [
                        f"[index {index}]",
                        f"score: {score:.4f}",
                        f"title: {doc.metadata.get('title', 'untitled')}",
                        f"doc_id: {doc.doc_id}",
                        f"content: {doc.content[:limit]}",
                    ]
                )
            )
        return "\n\n".join(serialized)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_query_for_compare(self, query: str) -> str:
        normalized = re.sub(r"[^\w\uAC00-\uD7A3]+", " ", query.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _normalize_next_query(self, candidate: str, original_query: str) -> str:
        cleaned_candidate = candidate.strip()
        if not cleaned_candidate:
            return original_query
        if self._normalize_query_for_compare(
            cleaned_candidate
        ) == self._normalize_query_for_compare(original_query):
            return original_query
        return cleaned_candidate

    async def _build_persona_bundle_node(self, state: GraphState) -> dict[str, Any]:
        profile = await memory_store.get_user_profile(state.session_id)
        session_summary = self._build_session_summary(state.chat_history)
        base_query = (state.next_query or state.query).strip() or state.query
        retrieval_query = await self._build_retrieval_query(
            base_query=base_query,
            chat_history=state.chat_history,
            session_summary=session_summary,
            profile=profile,
        )
        generation_hints = self._build_generation_hints(profile)
        context = await elasticsearch_store.hybrid_search(
            query=retrieval_query,
            k=settings.TOP_K_RESULTS,
            vector_weight=0.5,
        )
        return {
            "user_profile": profile,
            "session_summary": session_summary,
            "generation_hints": generation_hints,
            "retrieved_docs": list(context.documents),
            "retrieval_scores": list(context.scores),
            "retrieval_query": retrieval_query,
            "last_retrieval_query": retrieval_query,
        }

    async def _self_critique_node(self, state: GraphState) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        try:
            parsed = self._parse_json(
                await self._invoke(
                    self.selfrag_critique_prompt,
                    {
                        "query": state.query,
                        "retrieval_query": state.retrieval_query
                        or state.last_retrieval_query
                        or state.query,
                        "session_summary": state.session_summary
                        or "No recent conversation.",
                        "generation_hints": state.generation_hints
                        or "No special preferences provided.",
                        "documents": self._serialize_docs(
                            state.retrieved_docs, state.retrieval_scores, 420
                        )
                        or "No documents retrieved.",
                        "scores": json.dumps(state.retrieval_scores, ensure_ascii=False),
                    },
                )
            )
        except Exception:
            parsed = {}

        if state.retrieved_docs:
            fallback_answer = (
                "I could not produce a sufficiently grounded answer from the current "
                "documents. Another retrieval attempt is needed."
            )
        else:
            fallback_answer = (
                "No relevant documents were retrieved, so I cannot answer yet. "
                "I will try a more specific retrieval query."
            )

        answer = str(parsed.get("answer") or "").strip() or fallback_answer
        is_sufficient = (
            parsed.get("is_sufficient")
            if isinstance(parsed.get("is_sufficient"), bool)
            else False
        )
        next_query = self._normalize_next_query(
            str(parsed.get("next_query") or ""),
            state.query,
        )
        utility_score = max(
            1.0, min(5.0, self._safe_float(parsed.get("utility_score"), 1.0))
        )
        confidence = max(
            0.0, min(1.0, self._safe_float(parsed.get("confidence"), 0.0))
        )
        reasons = [
            str(reason)
            for reason in (parsed.get("insufficiency_reasons") or [])
            if str(reason).strip()
        ]

        selfrag_scores = SelfRagScores(
            utility_score=utility_score,
            confidence=confidence,
            insufficiency_reasons=reasons,
            next_query=next_query,
        )

        return {
            "answer": answer,
            "is_sufficient": is_sufficient,
            "next_query": next_query,
            "selfrag_scores": selfrag_scores,
        }

    async def _check_sufficiency_node(self, state: GraphState) -> dict[str, Any]:
        next_query = self._normalize_next_query(state.next_query, state.query)
        if state.is_sufficient:
            return {
                "next_action": "finalize",
                "next_query": next_query,
                "is_sufficient": True,
            }
        if state.loop_count < state.max_loops:
            return {
                "next_action": "retry",
                "next_query": next_query,
                "loop_count": state.loop_count + 1,
                "is_sufficient": False,
            }
        return {
            "next_action": "finalize",
            "next_query": next_query,
            "is_sufficient": False,
        }

    def get_graph(self) -> CompiledStateGraph | None:
        return self._graph

    async def prepare_state(
        self, query: str, session_id: str, chat_history: list[Message]
    ) -> GraphState:
        return GraphState(
            query=query,
            session_id=session_id,
            chat_history=chat_history,
            max_loops=settings.SELF_RAG_MAX_LOOPS,
        )

    async def close(self) -> None:
        try:
            await elasticsearch_store.close()
        except Exception:
            pass
        self._initialized = False


rag_graph = RAGGraph()
