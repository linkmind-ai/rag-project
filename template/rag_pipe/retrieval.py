import json
from datetime import datetime
from typing import Optional
import textwrap

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# from client import GeneratorClient
from prompts import BASE_PROMPT
from prompts import BASE_CHAT_PROMPT

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from FlagEmbedding import FlagReranker

with open("template/common/config.json", "r") as f:
    cfg = json.load(f)


def hybrid_search(query, vector_db, documents, k=2, bm25_weight=0.5):
    """
    Hybrid search combining BM25 and cosine similarity with customizable weighting

    Parameters:
    - query: 검색 쿼리
    - vector_db: 코사인 유사도 검색용 벡터 데이터베이스
    - documents: 원본 문서 리스트
    - k: 반환할 결과 수
    - bm25_weight: BM25 가중치 (0-1), 나머지는 코사인 유사도 가중치

    Returns:
    - 최종 결과 리스트
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["embedding"])
    reranker = FlagReranker(cfg["model"]["reranker"], use_fp16=True)
    
    # 1. 코사인 유사도 검색 (벡터 검색)
    cosine_results = vector_db.similarity_search_with_score(query, k=k*2)

    cosine_docs = [doc for doc, score in cosine_results]
    cosine_scores = [score for doc, score in cosine_results]

    max_score = max(cosine_scores)
    min_score = min(cosine_scores)
    score_range = max_score - min_score if max_score != min_score else 1
    normalized_cosine_scores = [1 - ((score - min_score) / score_range) for score in cosine_scores]

    # 2. BM25 준비 (Hugging Face tokenizer 기반)
    tokenized_docs = [tokenizer.tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenizer.tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = max(bm25_scores)
    min_bm25 = min(bm25_scores)
    bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
    normalized_bm25_scores = [(score - min_bm25) / bm25_range for score in bm25_scores]

    # 3. 하이브리드 점수 계산
    hybrid_scores = []
    for i, doc in enumerate(documents):
        cosine_score = 0
        for j, cosine_doc in enumerate(cosine_docs):
            if doc.page_content == cosine_doc.page_content:
                cosine_score = normalized_cosine_scores[j]
                break
        hybrid_score = (bm25_weight * normalized_bm25_scores[i]) + ((1 - bm25_weight) * cosine_score)
        hybrid_scores.append((doc, hybrid_score))

    ## 리랭커 미적용
    # hybrid_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k]
    # return [doc for doc, score in hybrid_results]
    
    # 후보군 (BM25 + Cosine top-k*3)
    hybrid_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k*3]
    candidate_docs = [doc for doc, score in hybrid_results]

    # -----------------------
    # 4. BGE 리랭커 적용
    # -----------------------
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    scores = reranker.compute_score(pairs)  # relevance score

    reranked_docs = [
        doc for _, doc in sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    ]

    # 최종 top-k 결과 반환
    return reranked_docs[:k]