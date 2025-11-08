# src/services/rag_agent.py

from langchain.prompts import PromptTemplate
from src.storage.elastic_store import ElasticStore
from src.clients.ollama_client import OllamaClient


class RagAgent:
    """HyDE와 RAG를 결합하여 사용자 쿼리에 답변하는 에이전트"""

    def __init__(self, vector_store: ElasticStore, llm_client: OllamaClient):
        self.vector_store = vector_store
        self.llm_client = llm_client

        self.hyde_prompt_template = PromptTemplate.from_template(
            "다음 question에 대한 요약된 답변을 작성해줘.\nquestion:\n{question}"
        )
        self.rag_prompt_template = PromptTemplate.from_template(
            "다음 question에 대해 context에 기반해서 답변해줘.\n"
            "단, 'context에 기반한 ~'과 같은 표현은 사용하지 마.\n"
            "question:\n{question}\n"
            "contexts:\n{contexts}"
        )

    def query(self, question: str) -> str | None:
        """전체 RAG 파이프라인(HyDE -> Search -> RAG)을 실행합니다."""

        print(f"--- 1. HyDE: '{question}'에 대한 가상 답변 생성 중...")
        hyde_prompt = self.hyde_prompt_template.format(question=question)
        expanded_query = self.llm_client.get_response(hyde_prompt)

        if not expanded_query:
            print("⚠️ HyDE 답변 생성 실패. 원본 질문으로 검색합니다.")
            expanded_query = question

        print("--- 2. Search: HyDE 답변으로 ES k-NN 검색 중...")
        contexts_docs = self.vector_store.search_knn(expanded_query, k=3)

        if not contexts_docs:
            print("❌ Elasticsearch에서 컨텍스트를 찾지 못했습니다.")
            return None

        print(f"--- 3. RAG: 검색된 컨텍스트(x{len(contexts_docs)})로 최종 답변 생성 중...")
        rag_prompt = self.rag_prompt_template.format(
            question=question,
            contexts=contexts_docs
        )
        final_answer = self.llm_client.get_response(rag_prompt)

        print("--- 4. RAG Pipeline 완료. ---")
        return final_answer