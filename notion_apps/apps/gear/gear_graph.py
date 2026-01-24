import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from models.state import GraphState, Message, Document
from stores.vector_store import elasticsearch_store
from gear import gear_retriever
from common.config import settings
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT


class RAGGraph:
    """랭그래프 기반 RAG 프로세스"""

    def __init__(self):
        self._llm: Optional[Ollama] = None
        self._graph = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT
        self.get_evidence_prompt = _GET_EVIDENCE_PROMPT 

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
        workflow.add_node("identify_evidence", self._identify_evidence_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "identify_evidence")
        workflow.add_edge("identify_evidence", END)

        self._graph = workflow.compile()

    async def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """검색 노드"""
        query = state.query
        use_gear = getattr(settings, 'USE_GEAR_RETRIEVAL', False)

        if use_gear:
            documents, gist_memory = await gear_retriever.gear_retrieve(
                query=query,
                max_steps=getattr(settings, 'GEAR_MAX_STEPS', 3),
                top_k=settings.TOP_K_RESULTS
            )
            return{
                "retrieved_docs": documents
            }
        
        else:
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

    def _build_context(self, retrieved_docs: List[Document]) -> str:
        """검색 문서 텍스트 변환"""
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        return context_text
    
    async def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        """응답 생성 노드"""
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
    
    async def _parse_evidence_response(self, response: str) -> List[int]:
        """LLM 응답 근거 인덱스 파싱"""
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*?\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response
                
            parsed = json.loads(json_str)

            if isinstance(parsed, dict) and "evidence_indices" in parsed:
                indices = parsed["evidence_indices"]
                if isinstance(indices, list):
                    return [int(idx) for idx in indices]
                
            if isinstance(parsed, list):
                return [int(idx) for idx in parsed]
            
            return []
        
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"근거 응답 파싱 오류: {e}")
            print(f"원본 응답: {response}")

            numbers = re.findall(r'\d+', response)
            if numbers:
                return [int(n) for n in numbers]
            
            return []
    
    async def _identify_evidence_node(self, state:GraphState) -> Dict[str, Any]:
        """답변 근거 문서 식별"""
        query = state.query
        answer = state.answer
        retrieved_docs = state.retrieved_docs

        if not retrieved_docs:
            return {"evidence_indices": []}
        
        documents_text = "\n\n".join([
            f"[인덱스 {i}]\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        prompt = self.get_evidence_prompt
        chain = prompt | self._llm

        input_data = {
            "query": query,
            "answer": answer,
            "documents": documents_text
        }

        try:
            result = await asyncio.to_thread(chain.invoke, input_data)

            evidence_indices = self._parse_evidence_response(result)

            valid_indices = [
                idx for idx in evidence_indices
                if 0 <= idx < len(retrieved_docs)
            ]
            return {"evidence_indices": valid_indices}
        
        except Exception as e:
            print(f"근거 문서 식별 오류: {e}")
            return {"evidence_indices": list(range(len(retrieved_docs)))}

    
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