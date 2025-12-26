import json
import os
from langchain.prompts import PromptTemplate
import ollama

from rag_prompts import *
from contents_to_vectordb import VectorDBConfig
from retrieval import HybridSearcher

    
config = VectorDBConfig.from_env_and_file()

# 쿼리 실행
question = input("질문을 입력하세요: ")

prompt = PromptTemplate.from_template(INITIAL_RESPONSE_PROMPT)

client = ollama.Client(host=config.ollama_host)

response = client.chat(
    model=config.generation_model_name,
    messages=[
        {
            'role': 'user',
            'content': prompt.format(question=question),
        },
    ]
)

expanded_query = response['message']['content']

searcher = HybridSearcher(
    es_host=config.es_host,
    es_id=config.es_id,
    es_api_key=config.es_api_key,
    index_name=config.es_index,
    embedding_model_name=config.embedding_model_name,
    reranker_model_name=config.reranker_model_name)

contexts = searcher.search(expanded_query, k=3)

# 최종 답변 출력
final_prompt = PromptTemplate.from_template(FINAL_RESPONSE_PROMPT)

final_response = client.chat(
    model=config.generation_model_name,
    messages=[
        {
            'role': 'user',
            'content': final_prompt.format(question=question, contexts=contexts),
        },
    ]
)
response = final_response['message']['content']

print(response)