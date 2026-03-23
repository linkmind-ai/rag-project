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
        retrieved_docs = result.get("retrieved_docs", [])
        retrieval_scores = result.get("retrieval_scores", [])

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", answer)

        elapsed_time = time.time() - start_time

        meta = {
            "selfrag_scores": self._normalize(result.get("selfrag_scores", {})),
            "loop_count": result.get("loop_count", 0),
            "is_sufficient": result.get("is_sufficient", False),
            "last_retrieval_query": result.get("last_retrieval_query", ""),
            "retrieval_scores": self._normalize(retrieval_scores),
        }

        logger.info(
            "[RAGService] process_query done session=%s docs=%s elapsed=%.2fs",
            session_id,
            len(retrieved_docs),
            elapsed_time,
        )

        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
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

                elif event_type == "on_chain_end" and name == "build_persona_bundle":
                    output = data.get("output", {})
                    retrieved_docs = output.get("retrieved_docs", [])
                    latest_meta["retrieval_scores"] = output.get("retrieval_scores", [])
                    latest_meta["last_retrieval_query"] = output.get(
                        "last_retrieval_query", ""
                    )
                    yield {
                        "type": "retrieve_end",
                        "message": f"Retrieved {len(retrieved_docs)} documents",
                        "doc_count": len(retrieved_docs),
                    }

                elif event_type == "on_chain_start" and name == "self_critique":
                    yield {
                        "type": "self_critique_start",
                        "message": "Generating answer and checking sufficiency",
                    }

                elif event_type == "on_chain_end" and name == "self_critique":
                    output = data.get("output", {})
                    full_response = output.get("answer", "") or full_response
                    scores = output.get("selfrag_scores")
                    if hasattr(scores, "model_dump"):
                        scores = scores.model_dump()
                    latest_meta["selfrag_scores"] = scores
                    latest_meta["is_sufficient"] = output.get("is_sufficient", False)
                    if full_response:
                        yield {"type": "content", "content": full_response}
                    yield {
                        "type": "self_critique_end",
                        "scores": scores,
                        "is_sufficient": output.get("is_sufficient", False),
                    }

                elif event_type == "on_chain_end" and name == "check_sufficiency":
                    output = data.get("output", {})
                    latest_meta["loop_count"] = output.get(
                        "loop_count", latest_meta.get("loop_count", 0)
                    )
                    latest_meta["is_sufficient"] = output.get(
                        "is_sufficient", latest_meta.get("is_sufficient", False)
                    )
                    if output.get("next_action") == "retry":
                        yield {
                            "type": "retry",
                            "loop_count": output.get("loop_count", 0),
                            "next_query": output.get("next_query", query),
                        }

            await memory_store.add_message(session_id, "user", query)
            await memory_store.add_message(session_id, "assistant", full_response)

            elapsed_time = time.time() - start_time
            yield {
                "type": "done",
                "full_response": full_response,
                "elapsed_time": elapsed_time,
                "sources": [
                    {
                        "index": idx,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "is_evidence": True,
                    }
                    for idx, doc in enumerate(retrieved_docs)
                ],
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
