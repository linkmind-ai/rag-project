import json
import asyncio
from typing import AsyncGenerator, Dict, Any
from fastapi.responses import StreamingResponse

from langgraph.state.chat_state import ChatState
from langgraph.cache.in_memory import InMemoryCache
from models.request import ChatRequest
from models.response import ChatStateResponse
from src.apps.models.basic_chat.base

syncredispool = syncredispool

class ChatService:

    def __init__(
            self,
            graph: Any,
            cache: InMemoryCache,
            semaphore: asyncio.Semaphore,
    ):
        self._graph = graph
        self._cache = cache
        self._semaphore = semaphore

    async def stream_chat(self, request:ChatRequest) -> StreamingResponse:
        rds = syncredispool.get_client()

        rds.setex(TAST_STATUS.format(task_id), REDIS_TTL, "stream")

        def event_stream() -> Generator[str, None, None]:
            state = self._cache.load_state(request.session_id)

            if not state.messages and request.history:
                for msg in request.history:
                    if msg.role == "user":
                        state.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        state.add_ai_message(msg.content)

                self._cache.save_state(request.session_id, state)
                yield f"data: {json.dumps({'type': 'status', 'state': 'restored'})}\n\n"
                rds.rpush(TASK_STATUS.format(task_id), json.dumps({'type':'status', 'state', 'restored'}))

            state.add_user_message(request.message, request.imgbase_64)

            config = {"configurable": {"thread_id": request.session_id}}
            last_status = None
            final_chat_state = None
            usage_emitted = False
            full_response = ""
            
            async for event in self._graph.astream_events(state, config, version = "v2"):
                kind = event["event"]

                if kind == "on_chat_model_steam":
                    node_name = event["metadata"].get("langgraph_node")

                    if node_name in ["generate"]:
                        chunk = event["data"]["chunk"]

                        if getattr(chunk, "content", None):
                            full_response += chunk.content
                            msg = {'type': 'token', 'content': chunk.content}
                            rds.rpush(CHAT_Q.format(task_id), json.dumps(msg, ensure_ascii=False))
                            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"



                    elif kind =="on_chain_start":
                        node_name = event.get("name", "")
                        if node_name in ["retrieve", "summarize"]:
                            msg = {'type': 'node_start', 'node': node_name}
                            rds.rpush(CHAT_STATUS.format(task_id), json.dumps(msg))
                            yield f"data: {json.dumps(msg)}\n\n"

                    elif kind =="on_chain_end":
                        node_name = event.get("name", "")
                        output = event["data"].get("output")

                        if isinstance(output, ChatState):
                            final_chat_state = output
                            if node_name == "emit_usage" and output.token_usage["total_tokens"] > 0 and not usage_emitted:
                                payload = {'type': 'usage', 'usage': output.token_usage}
                                rds.rpush(CHAT_Q.format(task_id), json.dumps(msg, ensure_ascii=False))
                                yield f"data: {json.dumps(payload)}\n\n"
                                usage_emitted = True

                            current_status = output.metadata.get("status")
                            if current_status and current_status != last_status:
                                msg = {"type": "status", "state": current_status}
                                rds.rpush(TASK_STATUS.format(task_id), json.dumps(msg))
                                yield f"data: {json.dumps(msg)}\n\n"
                                last_status = current_status

                    elif kind == "on_excution_end":
                        data = event.get("data", {})
                        output_container = data.get("state") or data.get("output")

                        if output_container is None:
                            continue

                        if hasattr(output_container, "values"):
                            maybe_state = output_container.values
                        elif isinstance(output_container, dict):
                            maybe_state = output_container.get("values")
                        else:
                            maybe_state = output_container

                        if isinstance(maybe_state, ChatState):
                            final_chat_state = maybe_state
                            if final_chat_state.token_usage["total_tokens"] > 0 and not usage_emitted:
                                payload = {"type": "usage", "usage": final_chat_state.token_usage}
                                rds.rpush(CHAT_Q.format(task_id), json.dumps(payload))
                                yield f"data: {json.dumps(payload)}\n\n"
                                usage_emitted = True

                if final_chat_state is None:
                    snapshot = self._graph.aget_state(config)
                    values = getattr(snapshot, "values", snapshot)
                    if isinstance(values, ChatState):
                        final_chat_state = values
                        if final_chat_state.token_usage["total_tokens"] > 0 and not usage_emitted:
                                
                                payload = {"type": "usage", "usage": final_chat_state.token_usage}
                                rds.rpush(CHAT_Q.format(task_id), json.dumps(payload))
                                yield f"data: {json.dumps(payload)}\n\n"
                                usage_emitted = True

                ref_payload = {"type": "references", "references": final_chat_state.references}

                if not re.search(r"주어진\s*정보로는", full_response) and final_chat_state.references:
                    rds.rpush(CHAT_Q.format(task_id), json.dumps(ref_payload, ensure_ascii=False))
                    yield f"data: {json.dumps(ref_payload, ensure_ascii = False)}\n\n"

                if final_chat_state is not None:
                    self._cache.save_state(request.session_id, final_chat_state)

                self._cache.reset_state(request.session_id, final_chat_state)

                rds.rpush(CHAT_Q.format(task_id), "data: [DONE]\n\n")
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type = "text/event_stream")
    
    async def get_state(self, session_id: str)-> ChatStateResponse:

        return await self._cache.dump_state(session_id)
    
