import asyncio
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from langgraph.graph import StateGraph, END
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from konlpy.tag import Mecab
import kss

from models.state import GraphState, Message, Document
from stores.vector_store import elasticsearch_store
from gear.gist_store import GistMemory
from common.config import settings
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT
from prompts.extract_triples_prompt import _EXTRACT_TRIPLES_PROMPT
from prompts.check_answerability_prompt import _CHECK_ANSWERABILITY_PROMPT
from prompts.rewrite_query_prompt import _REWRITE_QUERY_PROMPT


class GearRetriever:
    """기존 RAG 프로세스에 GeAR 모듈 추가"""

    def __init__(self):
        self._llm: Optional[Ollama] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._initialized = False
        self._initial_triples_cache: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self.mecab = Mecab()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT
        self.get_evidence_prompt = _GET_EVIDENCE_PROMPT
        self.extract_triples_prompt = _EXTRACT_TRIPLES_PROMPT
        self.check_answerability_prompt = _CHECK_ANSWERABILITY_PROMPT
        self.rewrite_query_prompt = _REWRITE_QUERY_PROMPT

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=1.0,
        )

        self._embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL, model=settings.EMBEDDING_MODEL
        )

        await self._extract_initial_triples()

        self._initialized = True

    async def _extract_triples_from_docs(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """문서에서 triples 추출"""
        triples = []

        # 문서 triples 생성 방법 수정 필요(한국어 외 타언어 triple 생성 로직 추가?)
        for doc in documents[:10]:
            content = doc.content
            sentences = kss.split_sentences(content)

            for sent in sentences[:3]:
                morphs = self.mecab.pos(sent)
                nouns = [w for w, t in morphs if t.startswith("NN")]
                verbs = [w for w, t in morphs if t.startswith("VV")]

                if len(nouns) < 2:
                    continue

                subject = nouns[0]
                object = nouns[-1]
                relation = verbs[0] if verbs else "related_to"

                triples.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "object": object,
                        "source": "initial",
                        "doc_id": doc.doc_id,
                    }
                )

        return triples

    async def _extract_initial_triples(self) -> None:
        """triples 추출 및 캐싱"""
        try:
            all_docs = await elasticsearch_store.get_all_documents_batch()

            if not all_docs:
                print("벡터스토어 연결 또는 문서 존재 여부 확인 필요")
                self._initial_triples_cache = []
                return

            print(f"전체 문서 {len(all_docs)}개에서 initial triples 추출 중")

            self._initial_triples_cache = await self._extract_triples_from_docs(
                all_docs
            )

            await self._save_initial_triples_cache()

        except Exception as e:
            print(f"Initial triples 추출 실패: {e}")
            self._initial_triples_cache = []

    def rrf_fusion(
        self, rank_lists: List[List[Document]], k: int = 60
    ) -> List[Document]:
        """RFF 알고리즘(랭크 역수 연산)"""
        scores = defaultdict(float)
        doc_map = {}

        for ranking in rank_lists:
            for rank, doc in enumerate(ranking, start=1):
                doc_id = doc.doc_id or id(doc)
                scores[doc_id] += 1 / (k + rank)
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    def _parse_triples_response(self, response: str) -> List[Dict[str, Any]]:
        """LLM 프롬프트 결과 triples 파싱"""
        try:
            json_match = re.search(r"\{.*?\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "triples" in parsed:
                    return parsed["triples"]

            return []
        except Exception as e:
            print(f"triples 파싱 오류: {e}")
            return []

    async def reader(
        self,
        passages: List[Document],
        query: str,
        existing_triples: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """문서에서 proximal triples 추출"""
        await self.initialize()

        passages_text = "\n\n".join(
            [f"[문서 {i+1}]\n{doc.content[:500]}" for i, doc in enumerate(passages[:5])]
        )

        existing_info = ""
        if existing_triples:
            existing_info = "\n기존 추출된 관계:\n" + "\n".join(
                [
                    f"- {t.get('subject', '')} -> {t.get('relation', '')} -> {t.get('object', '')}"
                    for t in existing_triples[:10]
                ]
            )

        prompt = self.extract_triples_prompt

        chain = prompt | self._llm

        try:
            result = await asyncio.to_thread(
                chain.invoke,
                {
                    "query": query,
                    "passages": passages_text,
                    "existing_info": existing_info,
                },
            )

            triples = self._parse_triples_response(result)
            return triples

        except Exception as e:
            print(f"triples 추출 오류: {e}")
            return []

    def _triple_to_text(self, triple: Dict[str, Any]) -> str:
        """triples 텍스트로 변환"""
        subject = triple.get("subject", "")
        relation = triple.get("relation", "")
        object = triple.get("object", "")
        return f"{subject} {relation} {object}"

    def _word_matching_score(
        self, triple1: Dict[str, Any], triple2: Dict[str, Any]
    ) -> float:
        """triple 간 키워드 기반 유사도 점수 매칭"""

        words1 = set()
        words2 = set()

        for key in ["subject", "relation", "object"]:
            text1 = str(triple1.get(key, "")).lower()
            text2 = str(triple2.get(key, "")).lower()

            words1.update(text1.split())
            words2.update(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 점수 매칭 기반"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        sim = np.dot(vec1, vec2) / (norm1 * norm2)

        return float((sim + 1) / 2)

    async def _self_attention_matching(
        self,
        proximal_triple: Dict[str, Any],
        proximal_text: str,
        initial_triples: List[Dict[str, Any]],
        threshold: float = 0.5,
        # 테스트로 임계치값 재설정
    ) -> Optional[Dict[str, Any]]:
        """셀프어텐션으로 initial triples 조회"""
        try:
            proximal_emb = await asyncio.to_thread(
                self._embeddings.embed_query, proximal_text
            )
            proximal_emb = np.array(proximal_emb)
        except Exception as e:
            print(f"proximal triples 임베딩 오류: {e}")
            return None

        best_match = None
        best_score = threshold

        for i_triple in initial_triples:
            i_text = self._triple_to_text(i_triple)

            try:
                i_emb = await asyncio.to_thread(self._embeddings.embed_query, i_text)
                i_emb = np.array(i_emb)
            except:
                continue

            word_score = self._word_matching_score(proximal_triple, i_triple)
            cosine_score = self._cosine_similarity(proximal_emb, i_emb)
            attention_score = 0.6 * word_score + 0.4 * cosine_score

            if attention_score > best_score:
                best_score = attention_score
                best_match = {
                    **i_triple,
                    "score": float(attention_score),
                    "word_score": float(word_score),
                    "cosine_score": float(cosine_score),
                }

        return best_match

    async def triple_link(
        self, proximal_triples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        proximal triples(LLM이 판단한 유효 triples)와 initial triples(리트리버 결과 triples) 링크 생성(매핑)
        키워드 매칭과 코사인 유사도 기반의 셀프어텐션 결합
        """

        if not self._initial_triples_cache:
            return proximal_triples

        print("triples 링크 시작: ")
        print(f"proximal triples: {len(proximal_triples)}개")
        print(f"_initial triples: {len(self._initial_triples_cache)}개")

        linked_triples = []

        for p_triple in proximal_triples:
            p_text = self._triple_to_text(p_triple)

            best_match = await self._self_attention_matching(
                p_triple, p_text, self._initial_triples_cache
            )

            linked_triple = {
                **p_triple,
                "linked_to": best_match if best_match else None,
                "link_score": best_match.get("score", 0.0) if best_match else 0.0,
            }

            linked_triples.append(linked_triple)

        print(f"연결완료: {sum(1 for t in linked_triples if t['linked_to'])}개 매칭")
        return linked_triples

    async def graph_expansion(
        self, triples: List[Dict[str, Any]], max_docs: int = 5
    ) -> List[Document]:
        """그래프 기반 검색 문서 확장"""

        if not triples:
            return []

        linked_triples = [
            t
            for t in triples
            if t.get("linked_to") and t.get("link_score", 0) > 0.5  # 스코어 임계값 수정
        ]

        if not linked_triples:
            linked_triples = triples[:3]

        expanded_triples = []

        for triple in linked_triples[:3]:
            subject = triple.get("subject", "")
            relation = triple.get("relation", "")
            object = triple.get("object", "")
            expanded_triple = f"{subject} {relation} {object}"
            expanded_triples.append(expanded_triple)

        all_docs = []

        for exp_triple in expanded_triples:
            context = await elasticsearch_store.hybrid_search(
                query=exp_triple, k=max_docs  # 여기에 사용자 질의도 더해본다면 어떨까?
            )
            all_docs.extend(context.documents)

        seen_ids = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.doc_id or id(doc)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        return unique_docs

    async def reason(
        self, triples: List[Dict[str, Any]], query: str
    ) -> Tuple[bool, str]:
        """생성된 triples로 답변 가능한지 판단"""
        # triples로 답변 가능한지 보다 expansion 결과 만들어진 서브그래프로 리트리버된 문서 기준으로 확인하는건?
        await self.initialize()

        if not triples:
            return False, "충분한 정보가 없습니다."

        triples_text = "\n".join(
            [
                f"- {t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
                for t in triples
            ]
        )

        prompt = self.check_answerability_prompt

        chain = prompt | self._llm

        try:
            result = await asyncio.to_thread(
                chain.invoke, {"query": query, "triples": triples_text}
            )

            json_match = re.search(r"\{.*?\}", result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                is_answerable = parsed.get("answerable", False)
                reasoning = parsed.get("reasoning", "")
                return is_answerable, reasoning

            return False, "현재 triples로 답변 불가능"

        except Exception as e:
            print(f"판단 가능여부 추론 오류: {e}")
            return False, str(e)

    async def rewrite_query(
        self, original_query: str, triples: List[Dict[str, Any]], reasoning: str
    ) -> str:
        """쿼리 재작성"""
        await self.initialize()

        triples_text = "\n".join(
            [
                f"- {t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
                for t in triples[-5:]
            ]
        )

        prompt = self.rewrite_query_prompt

        chain = prompt | self._llm

        try:
            result = await asyncio.to_thread(
                chain.invoke,
                {
                    "query": original_query,
                    "triples": triples_text,
                    "reasoning": reasoning,
                },
            )
            return result if result else original_query

        except Exception as e:
            print(f"쿼리 rewriting 오류: {e}")
            return original_query

    async def gear_retrieve(
        self, query: str, max_steps: int = 3, top_k: int = 10
    ) -> Tuple[List[Document], GistMemory]:
        """GeAR 파이프라인 실행"""
        await self.initialize()

        gist_memory = GistMemory()
        current_query = query
        step = 1
        all_retrived_passages = []

        print(f"Gear 검색 시작: {query}")

        while step <= max_steps:
            print(f"\n 리트리버 {step}/{max_steps} 번째 step")
            print(f"쿼리: {current_query}")

            base_context = await elasticsearch_store.hybrid_search(
                query=current_query, k=top_k
            )
            base_passages = base_context.documents
            print(f"리트리버 결과: {len(base_passages)}개 문서")

            if step == 1:
                proximal_triples = await self.reader(base_passages, query)
            else:
                proximal_triples = await self.reader(
                    base_passages, query, gist_memory.get_all_triples()
                )
            print(f"proximal triples 추출: {len(proximal_triples)}개")

            triples = await self.triple_link(proximal_triples)

            expanded_passages = await self.graph_expansion(
                triples, query, max_docs=top_k
            )
            print(f"그래프 확장 후 {len(expanded_passages)}개 문서")

            combined_passages = self.rrf_fusion([base_passages, expanded_passages])
            all_retrived_passages.append(combined_passages)

            gist_memory.add_triples(proximal_triples)

            is_answerable, reasoning = await self.reason(
                gist_memory.get_all_triples(), query
            )
            print(f"답변 가능 여부: {is_answerable}")
            print(f"답변 가능 여부 판단 근거: {reasoning}")

            if is_answerable:
                print(f"\n 답변 가능한 정보 수집 완료(step {step})")
                break
            else:
                if step < max_steps:
                    current_query = await self.rewrite_query(
                        query, gist_memory.get_all_triples(), reasoning
                    )
                    print(f"쿼리 재작성: {current_query}")
                step += 1

        final_passages = self.rrf_fusion(all_retrived_passages)

        print(f"\n최종결과: {len(final_passages)}개 문서")
        return final_passages[:top_k], gist_memory

    async def sync_ge_retreive(self, query: str, top_k: int = 5) -> List[Document]:
        """SyncGE 검색"""
        await self.initialize()

        print(f"SyncGE 검색: {query}")

        base_context = await elasticsearch_store.hybrid_search(query=query, k=top_k)
        base_passages = base_context.documents
        print(f"1. 기본 검색: {len(base_passages)}개")

        proximal_triples = await self.reader(base_passages, query)
        print(f"2. triples 추출: {len(proximal_triples)}개")

        triples = await self.triple_link(proximal_triples)

        expanded_passages = await self.graph_expansion(triples, query, max_docs=top_k)
        print(f"3. 그래프 확장: {len(expanded_passages)}개")

        combined_passages = self.rrf_fusion([base_passages, expanded_passages])

        print(f"4. 최종 근거 문서: {len(combined_passages[:top_k])}개")
        return combined_passages[:top_k]


gear_retriever = GearRetriever()
