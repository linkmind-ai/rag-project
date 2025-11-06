import json
from langchain.prompts import PromptTemplate
import ollama

from retrieval import hybrid_search
from prompts.rag_prompts import *

with open("template/common/config.json", "r") as f:
    cfg = json.load(f)

# 쿼리 실행
question = input("질문을 입력하세요: ")

prompt = PromptTemplate.from_template(INITIAL_RESPONSE_PROMPT)

client = ollama.Client(host=cfg["es"]["host"])

response = client.chat(
    model=cfg["model"]["generation"],
    messages=[
        {
            'role': 'user',
            'content': prompt.format(question=question),
        },
    ]
)

expanded_query = response['message']['content']

contexts = hybrid_search(expanded_query, loaded_vector_db, texts, k=3, bm25_weight=0.5)

# 최종 답변 출력
final_prompt = PromptTemplate.from_template(FINAL_RESPONSE_PROMPT)

final_response = client.chat(
    model=cfg["es"]["host"],
    messages=[
        {
            'role': 'user',
            'content': final_prompt.format(question=question, contexts=contexts),
        },
    ]
)
response = final_response['message']['content']

print(response)