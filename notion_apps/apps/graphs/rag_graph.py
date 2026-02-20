import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM  #  community → langchain_ollama로 변경
from langgraph.graph import END, StateGraph

from common.config import settings
from models.state import Document, GraphState
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT
from stores.vector_store import elasticsearch_store


class RAGGraph:
    """랭그래프 기반 RAG 프로세스"""

    def __init__(self):
        self._llm: Optional[OllamaLLM] = None
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

            #  OllamaLLM으로 변경 (langchain_ollama 패키지)
            self._llm = OllamaLLM(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=1.0
            )

            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        """그래프 구조 빌드"""
        #  TypedDict 기반 GraphState 그대로 사용 가능
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
        query = state["query"]

        context = await elasticsearch_store.hybrid_search(
            query=query,
            k=settings.TOP_K_RESULTS,
            vector_weight=0.5
        )

        return {
            "retrieved_docs": context.documents
        }

    def _build_context(self, retrieved_docs: List[Document]) -> str:
        """검색 문서 텍스트 변환"""
        context_text = "\n\n".join([
            f"Document {i+1}:\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        return context_text

    #  _prepare_messages 제거
    # chat_history가 이미 Sequence[BaseMessage]이므로 변환 불필요
    # 프롬프트에 chat_history를 그대로 전달하면 됨

    async def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        """응답 생성 노드"""
        query = state["query"]
        retrieved_docs: List[Document] = state.get("retrieved_docs", [])
        #  chat_history는 이미 Sequence[BaseMessage] - 변환 없이 그대로 사용
        chat_history: Sequence[BaseMessage] = state.get("chat_history", [])

        context_text = self._build_context(retrieved_docs)

        if chat_history:
            prompt = self.chat_with_history_prompt
            input_data = {
                "context": context_text,
                "history": chat_history,  #  BaseMessage 리스트 직접 전달
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
                json_str = json_match.group(0) if json_match else response

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
            return [int(n) for n in numbers] if numbers else []

    async def _identify_evidence_node(self, state: GraphState) -> Dict[str, Any]:
        """답변 근거 문서 식별"""
        query = state["query"]
        answer = state.get("answer", "")
        retrieved_docs: List[Document] = state.get("retrieved_docs", [])

        if not retrieved_docs:
            return {"evidence_indices": []}

        documents_text = "\n\n".join([
            f"[인덱스 {i}]\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])

        chain = self.get_evidence_prompt | self._llm
        input_data = {
            "query": query,
            "answer": answer,
            "documents": documents_text
        }

        try:
            result = await asyncio.to_thread(chain.invoke, input_data)
            evidence_indices = await self._parse_evidence_response(result)

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
        #  List[Message] → Sequence[BaseMessage]로 변경
        chat_history: Sequence[BaseMessage]
    ) -> GraphState:
        return {
            "query": query,
            "chat_history": chat_history,  #  add_messages reducer가 자동 병합 처리
            "session_id": session_id,
            #  TypedDict 필드 전체 명시 (retrieved_docs, answer, evidence_indices 기본값)
            "retrieved_docs": [],
            "answer": "",
            "evidence_indices": []
        }


rag_graph = RAGGraph()
