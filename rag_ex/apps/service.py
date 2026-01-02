import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from langchain_core.runnables import RunnableConfig

from models import GraphState
from rag_graph import rag_graph
from vector_store import elasticsearch_store
from memory_sotre import memory_store
from config import settings


class RAGService:
    """RAG시스템 서비스 레이어"""

    def __init__(self):
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """서비스 초기화"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            await rag_graph.initialize()
            self._initialized = True

    
    async def process_query(
            self,
            session_id: str,
            query: str,
            use_history: bool = True
    ) -> str:
        """쿼리-답변 동기 실행"""
        await self.initialize()

        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)

        initial_state = await rag_graph.prepare_state(
            query=query,
            session_id=session_id,
            chat_history=chat_history
        )

        graph = rag_graph.get_graph()

        result = await asyncio.to_thread(graph.invoke, initial_state)

        answer = result["answer"]

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", answer)

        return answer

    async def process_query_stream(
            self,
            session_id: str,
            query: str,
            use_history: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """쿼리-답변 스트리밍 실행"""
        await self.initialize()

        chat_history = []
        if use_history:
            chat_history = await memory_store.get_recent_messages(session_id)

        initial_state = await rag_graph.prepare_state(
            query=query,
            session_id=session_id ,
            chat_history=chat_history
        )

        config = RunnableConfig(
            configurable={
                "session_id": session_id,
                "thread_id": session_id # 추후 쓰레드ID는 수정 필요
            }
        )

        graph = rag_graph.get_graph()

        full_response = ""
        retrieved_docs = []

        async for event in graph.astream_events(initial_state, config, version="v1"):
            event_type = event.get("event")
            name = event.get("name", "")

            if event_type == "on_chain_start" and name == "retrieve":
                yield {
                    "type": "retrieve_start",
                    "message": "문서 검색 중..."
                }

            elif event_type == "on_chain_end" and name == "retrieve":
                data = event.get("data", {})
                output = data.get("output", {})
                retrieved_docs = output.get("retrieved_docs", [])

                yield {
                    "type": "retrieve_end",
                    "message": f"{len(retrieved_docs)}개의 문서를 찾았습니다.",
                    "doc_count": len(retrieved_docs)
                }

            elif event_type == "on_chain_start" and name == "generate":
                yield {
                    "type": "generate_start",
                    "message": "답변 생성 중..."
                }

            elif event_type == "on_chain_stream" and name == "generate":
                data = event.get("data", {})
                chunk = data.get("chunk", {})

                if isinstance(chunk, dict):
                    answer_chunk = chunk.get("answer", "")
                elif isinstance(chunk, str):
                    answer_chunk = chunk
                else:
                    answer_chunk = str(chunk) if chunk else ""

                if answer_chunk:
                    full_response += answer_chunk
                    yield {
                        "type": "content",
                        "content": answer_chunk
                    }

            elif event_type == "on_chain_end" and name == "generate":
                data = event.get("data", {})
                output = data.get("output", {})

                if not full_response and "answer" in output:
                    full_response = output["answer"]
                    yield {
                        "type": "content",
                        "content": full_response
                    }

                yield {
                    "type": "generate_end",
                    "content": "답변 생성 완료"
                }

        await memory_store.add_message(session_id, "user", query)
        await memory_store.add_message(session_id, "assistant", full_response)

        yield {
            "type": "done",
            "full_response": full_response
        }

    async def get_sources(
            self,
            query: str,
            top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        
        await self.initialize()

        k = top_k or settings.TOP_K_RESULTS

        context = await elasticsearch_store.hybrid_search(
            query=query,
            k=k
        )

        sources = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in zip(context.documents, context.scores)
        ]

        return {
            "sources": sources,
            "total_hits": context.total_hits
        }
    
rag_service = RAGService()