# clients/retriever_client.py

from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker


class HybridRetriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("jhgan/ko-sbert-sts")
        self.reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)

    def hybrid_search(self, query, vector_db, all_documents, k=5, bm25_weight=0.5):
        # ------------------------------
        # 1. 벡터 코사인 유사도 검색
        # ------------------------------
        cosine_results = vector_db.similarity_search_with_score(query, k=k*2)

        cosine_docs = [doc for doc, score in cosine_results]
        cosine_scores = [score for _, score in cosine_results]

        # 정규화
        max_score = max(cosine_scores)
        min_score = min(cosine_scores)
        range_score = max_score - min_score if max_score != min_score else 1
        norm_cosine = [1 - ((s - min_score) / range_score) for s in cosine_scores]

        # ------------------------------
        # 2. BM25
        # ------------------------------
        tokenized_docs = [self.tokenizer.tokenize(d.page_content) for d in all_documents]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        max_b = max(bm25_scores)
        min_b = min(bm25_scores)
        range_b = max_b - min_b if max_b != min_b else 1
        norm_bm25 = [(s - min_b) / range_b for s in bm25_scores]

        # ------------------------------
        # 3. 하이브리드 스코어 결합
        # ------------------------------
        hybrid_scores = []
        for i, doc in enumerate(all_documents):
            cosine_score = 0
            for j, cosine_doc in enumerate(cosine_docs):
                if doc.page_content == cosine_doc.page_content:
                    cosine_score = norm_cosine[j]
                    break
            final_score = bm25_weight * norm_bm25[i] + (1 - bm25_weight) * cosine_score
            hybrid_scores.append((doc, final_score))

        # 후보 k*3
        hybrid_candidates = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:k*3]
        candidate_docs = [d for d, _ in hybrid_candidates]

        # ------------------------------
        # 4. BGE reranker
        # ------------------------------
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.compute_score(pairs)

        reranked_docs = [
            d for _, d in sorted(
                zip(scores, candidate_docs),
                key=lambda x: x[0],
                reverse=True
            )
        ]

        return reranked_docs[:k]
