import asyncio
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from models import GraphState, Message
from vector_store import elasticsearch_store
from config import settings
from prompt import _CHAT_PROMPT, _CHAT_WITH_HISTORY_PROMPT


class RAGGraph:
    """랭그래프 기반 RAG 프로세스"""

    def __init__(self):
        self._llm: Optional[Ollama] = None
        self._graph = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT

    async def initialize(self) -> None:
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
        
            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=1.0
            )

            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        """그래프 구조 빌드"""
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        workflow.set_entry_point("retreive")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        self._graph = workflow.compile()

    async def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """검색 노드"""
        query = state.query

        context = await elasticsearch_store.hybrid_search(
            query=query,
            k=settings.TOP_K_RESULTS,
            vector_weight=0.5
        )

        return {
            "retrieved_docs": context.documents
        }
    
    def _prepare_messages(self, chat_history: List[Message]) -> List[Any]:
        """대화 이력 메시지 변환"""
        messages = []
        for msg in chat_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
            else:
                messages.append(SystemMessage(content=msg.content))

    def _build_context(self, retrieved_docs: List[Any]) -> str:
        """검색 문서 텍스트 변환"""
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        return context_text
    
    async def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        """노드 """
        query = state.query
        retrieved_docs = state.retrieved_docs
        chat_history = state.chat_history

        context_text = self._build_context(retrieved_docs)

        if chat_history:
            messages = self._prepare_messages(chat_history)
            prompt = self.chat_with_history_prompt
            input_data = {
                "context": context_text,
                "history": messages,
                "query": query
            }

        else:
            prompt = self.chat_prompt
            input_data = {
                "context": context_text,
                "query": query
            }


        chain = prompt | self._llm

        answer = await asyncio.to_thread(chain.invoke, input_data)

        return {
            "answer": answer
        }
    
    def get_graph(self):
        """컴파일된 그래프 반환"""
        return self._graph
    
    async def prepare_state(
            self,
            query: str,
            session_id: str,
            chat_history: List[Message]
    ) -> GraphState:
        return GraphState(
            query=query,
            chat_history=chat_history,
            session_id=session_id
        )
    

rag_graph = RAGGraph()