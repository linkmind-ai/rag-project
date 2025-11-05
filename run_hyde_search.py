# run_hyde_search.py

import config
import ollama
from langchain.prompts import PromptTemplate
import elasticsearch_manager
import search_pipeline  # 새로 만든 ES 검색 파이프라인


def run_hyde_pipeline(question, es_client, tokenizer, model):
    """HyDE와 RAG(Elasticsearch)를 사용하여 질문에 답변합니다."""

    print(f"\n--- 원본 질문 ---\n{question}\n")

    # 1. HyDE: 가상 답변 생성 (Ollama 사용)
    initial_response_template = """
        다음 question에 대한 요약된 답변을 작성해줘.
        question:
        {question}
        """
    prompt = PromptTemplate.from_template(initial_response_template)

    try:
        client = ollama.Client(host=config.OLLAMA_HOST)
        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt.format(question=question)}]
        )
        expanded_query = response['message']['content']
        print(f"--- HyDE 가상 답변 (검색용) ---\n{expanded_query}\n")
    except Exception as e:
        print(f"❌ Ollama (HyDE) 호출 오류: {e}")
        print("가상 답변 생성 실패. 원본 질문으로 검색합니다.")
        expanded_query = question

    # 2. 가상 답변을 임베딩
    try:
        query_vector = elasticsearch_manager.embed_text(expanded_query, tokenizer, model)
    except Exception as e:
        print(f"❌ 쿼리 임베딩 오류: {e}")
        return None  # 👈 [추가] 오류 시 None 반환

    # 3. Elasticsearch k-NN 검색 (ChromaDB 대체)
    contexts_docs = search_pipeline.search_es_knn(
        es_client=es_client,
        query_vector=query_vector,
        k=3
    )

    if not contexts_docs:
        print("❌ Elasticsearch에서 검색된 컨텍스트가 없습니다.")
        return None  # 👈 [추가] 오류 시 None 반환

    print(f"--- 검색된 상위 {len(contexts_docs)}개 컨텍스트 ---")
    for i, doc in enumerate(contexts_docs):
        print(f"[{i + 1}] (Title: {doc.metadata.get('title')})\n{doc.page_content[:150]}...\n")

    # 4. 최종 답변 생성 (Ollama RAG)
    final_response_template = """
        다음 question에 대해 context에 기반해서 답변해줘. 
        단, 'context에 기반한 ~'과 같은 표현은 사용하지 마.
        question:
        {question}
        contexts:
        {contexts}
        """
    final_prompt = PromptTemplate.from_template(final_response_template)

    try:
        final_response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {'role': 'user', 'content': final_prompt.format(question=question, contexts=contexts_docs)}
            ]
        )
        response = final_response['message']['content']
        print(f"--- 최종 답변 ---\n{response}")

        # 📌 [수정] 계산된 response 값을 반환(return)해야 합니다.
        return response

    except Exception as e:
        print(f"❌ OLLAMA (RAG) 호출 오류: {e}")
        return None  # 👈 [추가] 오류 시 None 반환


if __name__ == "__main__":
    print("--- 1. ES RAG 파이프라인 시작 ---")

    # 1. Elasticsearch 클라이언트 가져오기
    es_client = elasticsearch_manager.get_es_client()

    # 2. 임베딩 모델 로드하기
    tokenizer, model = elasticsearch_manager.get_embedding_model()

    if es_client and tokenizer and model:
        print("\n--- 2. RAG 파이프라인 실행 (Q1) ---")
        question_1 = "AI의 가치 정렬(value alignment) 문제는 구체적으로 어떤 우려를 말하나요?"
        run_hyde_pipeline(question_1, es_client, tokenizer, model)

        print("\n--- 3. RAG 파이프라인 실행 (Q2) ---")
        question_2 = "단일성 정체감의 장애 현상은 어떻게 나타나는가?"
        run_hyde_pipeline(question_2, es_client, tokenizer, model)
    else:
        print("❌ ES 클라이언트 또는 임베딩 모델 로드에 실패했습니다. config.py를 확인하세요.")