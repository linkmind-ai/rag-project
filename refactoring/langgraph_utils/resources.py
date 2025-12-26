import ollama
from rag_prompts import *
from contents_to_vectordb import VectorDBConfig
from retrieval import HybridSearcher
from langchain.prompts import PromptTemplate

config = VectorDBConfig.from_env_and_file()

client = ollama.Client(host=config.ollama_host)

hyde_prompt = PromptTemplate.from_template(INITIAL_RESPONSE_PROMPT)
final_prompt = PromptTemplate.from_template(FINAL_RESPONSE_PROMPT)

searcher = HybridSearcher(
    es_host=config.es_host,
    es_id=config.es_id,
    es_api_key=config.es_api_key,
    index_name=config.es_index,
    embedding_model_name=config.embedding_model_name,
    reranker_model_name=config.reranker_model_name,
)