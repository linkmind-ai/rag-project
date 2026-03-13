import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from graphs.rag_graph import rag_graph
from langchain_core.runnables import RunnableConfig
from stores.memory_store import memory_store

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self) -> None:
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            await rag_graph.initialize()
            self._initialized = True

    async def process_query(
        self, session_id: str, query: str, use_history: bool = True
    ) -> dict[str, Any]:
        start_time = time.time()
        await self.initialize()

        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)

        initial_state = await rag_graph.prepare_state(
            query=query,
            session_id=session_id,
            chat_history=chat_history,
        )

        graph = rag_graph.get_graph()
        result = await graph.ainvoke(initial_state.model_dump())

        answer = result.get("answer", "")
        evidence_indices = result.get("evidence_indices", [])
        retrieved_docs = result.get("retrieved_docs", [])

        evidence_docs = [
            retrieved_docs[idx]
            for idx in evidence_indices
            if isinstance(idx, int) and 0 <= idx < len(retrieved_docs)
        ]

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", answer)

        elapsed_time = time.time() - start_time

        meta = {
            "transparency": self._normalize(result.get("transparency", {})),
            "route": self._normalize(result.get("route", {})),
            "selfrag_scores": self._normalize(result.get("selfrag_scores", {})),
            "loop_count": result.get("loop_count", 0),
            "is_sufficient": result.get("is_sufficient", False),
            "last_retrieval_query": result.get("last_retrieval_query", ""),
        }

        logger.info(
            "[RAGService] process_query done session=%s docs=%s evidence=%s elapsed=%.2fs",
            session_id,
            len(retrieved_docs),
            len(evidence_indices),
            elapsed_time,
        )

        return {
            "answer": answer,
            "evidence_indices": evidence_indices,
            "evidence_docs": evidence_docs,
            "all_docs": retrieved_docs,
            "elapsed_time": elapsed_time,
            "meta": meta,
        }

    def _normalize(self, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, dict):
            return {k: self._normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize(v) for v in value]
        return value

    async def process_query_stream(
        self, session_id: str, query: str, use_history: bool = True
    ) -> AsyncGenerator[dict[str, Any], None]:
        start_time = time.time()
        try:
            await self.initialize()

            chat_history = []
            if use_history:
                chat_history = await memory_store.get_recent_messages(session_id)

            initial_state = await rag_graph.prepare_state(
                query=query,
                session_id=session_id,
                chat_history=chat_history,
            )

            config = RunnableConfig(
                configurable={"session_id": session_id, "thread_id": session_id}
            )

            graph = rag_graph.get_graph()
            full_response = ""
            retrieved_docs = []
            evidence_indices = []
            latest_meta: dict[str, Any] = {}

            async for event in graph.astream_events(
                initial_state.model_dump(), config, version="v1"
            ):
                event_type = event.get("event")
                name = event.get("name", "")
                data = event.get("data", {})

                if event_type == "on_chain_start" and name == "build_persona_bundle":
                    yield {
                        "type": "retrieve_start",
                        "message": "Document retrieval started",
                    }
                    yield {
                        "type": "persona_start",
                        "message": "Building persona evidence bundle",
                    }

                elif event_type == "on_chain_end" and name == "build_persona_bundle":
                    output = data.get("output", {})
                    retrieved_docs = output.get("retrieved_docs", [])
                    yield {
                        "type": "retrieve_end",
                        "message": f"Retrieved {len(retrieved_docs)} documents",
                        "doc_count": len(retrieved_docs),
                    }
                    yield {"type": "persona_end", "doc_count": len(retrieved_docs)}

                elif event_type == "on_chain_start" and name == "generate_draft":
                    yield {"type": "generate_start", "message": "Generating draft"}

                elif event_type == "on_chain_end" and name == "generate_draft":
                    output = data.get("output", {})
                    full_response = output.get("answer", "") or output.get(
                        "draft_answer", ""
                    )
                    if full_response:
                        yield {"type": "content", "content": full_response}
                    yield {
                        "type": "generate_end",
                        "message": "Draft generation completed",
                    }

                elif event_type == "on_chain_end" and name == "self_critique":
                    output = data.get("output", {})
                    scores = output.get("selfrag_scores")
                    if hasattr(scores, "model_dump"):
                        scores = scores.model_dump()
                    latest_meta["selfrag_scores"] = scores
                    yield {"type": "self_critique_end", "scores": scores}

                elif event_type == "on_chain_start" and name == "reinforce_retrieve":
                    yield {
                        "type": "reinforce_start",
                        "message": "Evidence reinforcement triggered",
                    }

                elif event_type == "on_chain_end" and name == "reinforce_retrieve":
                    output = data.get("output", {})
                    loop_count = output.get("loop_count")
                    latest_meta["loop_count"] = loop_count
                    yield {"type": "reinforce_end", "loop_count": loop_count}

                elif event_type == "on_chain_end" and name == "finalize_response":
                    output = data.get("output", {})
                    full_response = output.get("answer", full_response)
                    transparency = output.get("transparency", {})
                    latest_meta["transparency"] = transparency
                    yield {"type": "finalize", "transparency": transparency}

                elif event_type == "on_chain_start" and name == "identify_evidence":
                    yield {"type": "evidence_start", "message": "Identifying evidence"}

                elif event_type == "on_chain_end" and name == "identify_evidence":
                    output = data.get("output", {})
                    evidence_indices = output.get("evidence_indices", [])
                    docs = output.get("retrieved_docs", retrieved_docs)
                    evidence_docs = [
                        {
                            "index": idx,
                            "content": docs[idx].content,
                            "metadata": docs[idx].metadata,
                        }
                        for idx in evidence_indices
                        if isinstance(idx, int) and 0 <= idx < len(docs)
                    ]
                    yield {
                        "type": "evidence_end",
                        "message": f"Found {len(evidence_indices)} evidence documents",
                        "evidence_indices": evidence_indices,
                        "evidence_docs": evidence_docs,
                    }

            await memory_store.add_message(session_id, "user", query)
            await memory_store.add_message(session_id, "assistant", full_response)

            elapsed_time = time.time() - start_time
            yield {
                "type": "done",
                "full_response": full_response,
                "evidence_indices": evidence_indices,
                "elapsed_time": elapsed_time,
                "meta": latest_meta,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                "[RAGService] stream error session=%s error=%s elapsed=%.2fs",
                session_id,
                str(e),
                elapsed_time,
            )
            yield {
                "type": "error",
                "message": f"Processing error: {e}",
                "error_detail": str(e),
                "elapsed_time": elapsed_time,
            }

    async def submit_feedback(self, feedback: dict[str, Any]) -> dict[str, Any]:
        session_id = str(feedback.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("session_id is required")

        event = await memory_store.add_feedback_event(session_id, feedback)
        profile = await memory_store.update_profile_from_feedback(session_id, feedback)
        return {
            "success": True,
            "session_id": session_id,
            "feedback_event": event,
            "updated_profile": profile,
        }


rag_service = RAGService()
