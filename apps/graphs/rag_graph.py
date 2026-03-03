from __future__ import annotations

import asyncio
import json
import re
from types import TracebackType
from typing import Any

from common.config import settings
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from models.state import (
    Document,
    EvidenceBundleE0,
    GlobalMessagePoolM0,
    GraphState,
    Message,
    RouteDecision,
    SelfRagScores,
)
from prompts.chat_history_prompt import _CHAT_WITH_HISTORY_PROMPT
from prompts.chat_prompt import _CHAT_PROMPT
from prompts.get_evidence_prompt import _GET_EVIDENCE_PROMPT
from prompts.persona_contextual_retrieval_prompt import (
    _PERSONA_CONTEXTUAL_RETRIEVAL_PROMPT,
)
from prompts.persona_rerank_prompt import _PERSONA_RERANK_PROMPT
from prompts.route_query_prompt import _ROUTE_QUERY_PROMPT
from prompts.selfrag_critique_prompt import _SELFRAG_CRITIQUE_PROMPT
from prompts.selfrag_draft_prompt import _SELFRAG_DRAFT_PROMPT
from prompts.selfrag_rewrite_prompt import _SELFRAG_REWRITE_PROMPT
from prompts.transparency_prompt import _TRANSPARENCY_PROMPT
from stores.memory_store import memory_store
from stores.vector_store import elasticsearch_store

_STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "are"}


class RAGGraph:
    def __init__(self) -> None:
        self._llm: Ollama | None = None
        self._graph: CompiledStateGraph | None = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self.chat_prompt = _CHAT_PROMPT
        self.chat_with_history_prompt = _CHAT_WITH_HISTORY_PROMPT
        self.get_evidence_prompt = _GET_EVIDENCE_PROMPT
        self.route_query_prompt = _ROUTE_QUERY_PROMPT
        self.persona_contextual_prompt = _PERSONA_CONTEXTUAL_RETRIEVAL_PROMPT
        self.persona_rerank_prompt = _PERSONA_RERANK_PROMPT
        self.selfrag_draft_prompt = _SELFRAG_DRAFT_PROMPT
        self.selfrag_critique_prompt = _SELFRAG_CRITIQUE_PROMPT
        self.selfrag_rewrite_prompt = _SELFRAG_REWRITE_PROMPT
        self.transparency_prompt = _TRANSPARENCY_PROMPT

    async def __aenter__(self) -> "RAGGraph":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._lock:
            if self._initialized:
                return
            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=0.4,
            )
            self._build_graph()
            self._initialized = True

    def _build_graph(self) -> None:
        wf = StateGraph(GraphState)
        wf.add_node("route_input", self._route_input_node)
        wf.add_node("build_persona_bundle", self._build_persona_bundle_node)
        wf.add_node("generate_draft", self._generate_node)
        wf.add_node("self_critique", self._self_critique_node)
        wf.add_node("check_sufficiency", self._check_sufficiency_node)
        wf.add_node("reinforce_retrieve", self._reinforce_retrieve_node)
        wf.add_node("finalize_response", self._finalize_response_node)
        wf.add_node("identify_evidence", self._identify_evidence_node)
        wf.set_entry_point("route_input")
        wf.add_edge("route_input", "build_persona_bundle")
        wf.add_edge("build_persona_bundle", "generate_draft")
        wf.add_edge("generate_draft", "self_critique")
        wf.add_edge("self_critique", "check_sufficiency")
        wf.add_conditional_edges(
            "check_sufficiency",
            self._route_after_sufficiency,
            {"reinforce": "reinforce_retrieve", "finalize": "finalize_response"},
        )
        wf.add_edge("reinforce_retrieve", "generate_draft")
        wf.add_edge("finalize_response", "identify_evidence")
        wf.add_edge("identify_evidence", END)
        self._graph = wf.compile()

    def _route_after_sufficiency(self, state: GraphState) -> str:
        return state.next_action

    async def _invoke(self, prompt: Any, data: dict[str, Any]) -> str:
        return str(await asyncio.to_thread((prompt | self._llm).invoke, data))

    def _parse_json(self, text: str) -> dict[str, Any]:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        s = m.group(1) if m else text
        if not m:
            o = re.search(r"\{.*\}", text, re.DOTALL)
            if o:
                s = o.group(0)
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    def _serialize_docs(self, docs: list[Document], limit: int = 500) -> str:
        return "\n\n".join(
            [
                f"[인덱스 {i}] [index {i}]\ntitle: {d.metadata.get('title','untitled')}\ndoc_id: {d.doc_id}\ncontent: {d.content[:limit]}"
                for i, d in enumerate(docs)
            ]
        )

    def _extract_keywords(self, text: str) -> set[str]:
        t = re.findall(r"[\uac00-\ud7a3a-zA-Z0-9]+", text.lower())
        return {x for x in t if len(x) >= 2 and x not in _STOPWORDS}

    def _calculate_keyword_overlap(
        self, answer: str, doc_content: str
    ) -> tuple[float, set[str]]:
        a = self._extract_keywords(answer)
        d = self._extract_keywords(doc_content)
        if not a:
            return 0.0, set()
        m = a & d
        return len(m) / len(a), m

    async def _route_input_node(self, state: GraphState) -> dict[str, Any]:
        q = state.query.lower()
        task, risk, policy, used = "ambiguous", "low", "adaptive", False
        if any(k in q for k in ("story", "poem", "창작", "소설", "브레인스토밍")):
            task, policy = "creative", "minimal"
        elif any(k in q for k in ("hello", "hi", "안녕", "반가워")):
            task, policy = "conversational", "minimal"
        elif any(
            k in q
            for k in ("medical", "legal", "finance", "의료", "법률", "투자", "세금")
        ):
            task, risk, policy = "factual", "high", "forced"
        elif any(
            k in q
            for k in ("fact", "source", "date", "정확", "사실", "근거", "출처", "최신")
        ):
            task, policy = "factual", "forced"
        if task == "ambiguous" and settings.ROUTING_GATE_MODE == "rule_llm_fallback":
            try:
                p = self._parse_json(
                    await self._invoke(self.route_query_prompt, {"query": state.query})
                )
                task = (
                    p.get("task_type", task)
                    if p.get("task_type")
                    in {"creative", "conversational", "factual", "ambiguous"}
                    else task
                )
                risk = (
                    p.get("risk_level", risk)
                    if p.get("risk_level") in {"low", "high"}
                    else risk
                )
                policy = (
                    p.get("retrieval_policy", policy)
                    if p.get("retrieval_policy") in {"minimal", "forced", "adaptive"}
                    else policy
                )
                used = True
            except Exception:
                pass
        return {
            "route": RouteDecision(
                task_type=task,
                risk_level=risk,
                retrieval_policy=policy,
                used_llm_fallback=used,
            ),
            "max_loops": settings.SELF_RAG_MAX_LOOPS,
        }

    async def _build_persona_bundle_node(self, state: GraphState) -> dict[str, Any]:
        profile = await memory_store.get_user_profile(state.session_id)
        session_summary = " | ".join(
            [
                f"{m.role}:{m.content[:80].replace(chr(10),' ')}"
                for m in state.chat_history[-6:]
            ]
        )
        rq = state.query
        plan: dict[str, Any] = {}
        try:
            p = self._parse_json(
                await self._invoke(
                    self.persona_contextual_prompt,
                    {
                        "query": state.query,
                        "profile": json.dumps(profile, ensure_ascii=False),
                        "session_summary": session_summary,
                    },
                )
            )
            rq = p.get("rewritten_query") or rq
            plan = {
                "source_plan": p.get("source_plan", []),
                "notes": p.get("notes", ""),
            }
        except Exception:
            pass
        k = (
            settings.TOP_K_RESULTS
            if state.route.retrieval_policy != "minimal"
            else max(1, min(2, settings.TOP_K_RESULTS))
        )
        docs = list(
            (
                await elasticsearch_store.hybrid_search(
                    query=rq, k=k, vector_weight=0.5
                )
            ).documents
        )
        rerank_notes: list[str] = []
        if docs:
            try:
                p = self._parse_json(
                    await self._invoke(
                        self.persona_rerank_prompt,
                        {
                            "query": state.query,
                            "profile": json.dumps(profile, ensure_ascii=False),
                            "documents": self._serialize_docs(docs, 280),
                        },
                    )
                )
                idxs = [
                    i
                    for i in (p.get("ranked_indices") or [])
                    if isinstance(i, int) and 0 <= i < len(docs)
                ]
                if idxs:
                    rem = [i for i in range(len(docs)) if i not in set(idxs)]
                    docs = [docs[i] for i in (idxs + rem)]
                rerank_notes = [str(n) for n in (p.get("rerank_notes") or [])]
            except Exception:
                rerank_notes = ["persona rerank fallback used"]
        summaries = [
            {
                "doc_index": i,
                "summary": d.content[:180],
                "citation": d.metadata.get("title", d.doc_id),
                "confidence": 0.5,
            }
            for i, d in enumerate(docs)
        ]
        citations = [
            {
                "doc_index": i,
                "citation": s["citation"],
                "title": docs[i].metadata.get("title", "untitled"),
            }
            for i, s in enumerate(summaries)
        ]
        bundle = EvidenceBundleE0(
            top_docs=docs, doc_summaries=summaries, citations_meta=citations
        )
        gmp = GlobalMessagePoolM0(
            profile_snapshot=profile,
            session_summary=session_summary,
            retrieval_plan={
                "original_query": state.query,
                "rewritten_query": rq,
                **plan,
            },
            rerank_notes=rerank_notes,
        )
        return {
            "retrieved_docs": docs,
            "persona_bundle": bundle,
            "global_message_pool": gmp,
            "last_retrieval_query": rq,
        }

    def _prepare_messages(self, chat_history: list[Message]) -> list[Any]:
        out: list[Any] = []
        for m in chat_history:
            out.append(
                HumanMessage(content=m.content)
                if m.role == "user"
                else (
                    AIMessage(content=m.content)
                    if m.role == "assistant"
                    else SystemMessage(content=m.content)
                )
            )
        return out

    def _build_context(self, docs: list[Document]) -> str:
        return "\n\n".join(
            [f"Document {i+1}:\n{d.content}" for i, d in enumerate(docs)]
        )

    async def _generate_node(self, state: GraphState) -> dict[str, Any]:
        docs = state.persona_bundle.top_docs or state.retrieved_docs
        draft = ""
        try:
            draft = (
                await self._invoke(
                    self.selfrag_draft_prompt,
                    {
                        "query": state.query,
                        "context": self._build_context(docs),
                        "documents": self._serialize_docs(docs, 420),
                        "doc_summaries": json.dumps(
                            state.persona_bundle.doc_summaries, ensure_ascii=False
                        ),
                        "global_pool": json.dumps(
                            state.global_message_pool.model_dump(), ensure_ascii=False
                        ),
                    },
                )
            ).strip()
        except Exception:
            draft = ""
        if not draft:
            context = self._build_context(docs)
            if state.chat_history:
                draft = str(
                    await asyncio.to_thread(
                        (self.chat_with_history_prompt | self._llm).invoke,
                        {
                            "context": context,
                            "history": self._prepare_messages(state.chat_history),
                            "query": state.query,
                        },
                    )
                ).strip()
            else:
                draft = str(
                    await asyncio.to_thread(
                        (self.chat_prompt | self._llm).invoke,
                        {"context": context, "query": state.query},
                    )
                ).strip()
        return {"draft_answer": draft, "answer": draft}

    async def _self_critique_node(self, state: GraphState) -> dict[str, Any]:
        answer = state.draft_answer or state.answer
        docs = state.persona_bundle.top_docs or state.retrieved_docs
        segs = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()] or (
            [answer.strip()] if answer.strip() else []
        )
        parsed = {}
        if segs and docs:
            try:
                parsed = self._parse_json(
                    await self._invoke(
                        self.selfrag_critique_prompt,
                        {
                            "query": state.query,
                            "draft_answer": answer,
                            "segments": json.dumps(segs, ensure_ascii=False),
                            "documents": self._serialize_docs(docs, 420),
                        },
                    )
                )
            except Exception:
                parsed = {}
        rel = [
            float(x)
            for x in (parsed.get("rel_scores") or [])
            if isinstance(x, (int, float))
        ]
        if not rel:
            rel = [
                min(
                    1.0,
                    max(
                        [self._calculate_keyword_overlap(s, d.content)[0] for d in docs]
                        or [0.0]
                    )
                    * 1.5,
                )
                for s in segs
            ]
        delta = (
            settings.SELF_RAG_LOOP0_DELTA
            if state.strictness_level <= 0
            else (
                settings.SELF_RAG_LOOP1_DELTA
                if state.strictness_level == 1
                else settings.SELF_RAG_LOOP2_DELTA
            )
        )
        retrieve_decisions = [
            str(x) for x in (parsed.get("retrieve_decisions") or [])
        ] or ["Yes" if r < delta else "No" for r in rel]
        labels = [str(x).lower() for x in (parsed.get("support_labels") or [])]
        if not labels:
            labels = [
                (
                    "fully"
                    if r >= max(0.2, delta - 0.15)
                    else "partial" if r >= 0.08 else "none"
                )
                for r in rel
            ]
        util = float(parsed.get("utility_score", 0.0) or 0.0)
        util = max(
            1.0,
            min(
                5.0,
                util if util > 0 else 1.0 + (sum(rel) / len(rel) if rel else 0.0) * 4.0,
            ),
        )
        n = max(1, len(labels))
        full = sum(1 for x in labels if x == "fully") / n
        none = sum(1 for x in labels if x == "none") / n
        partial_or_no = sum(1 for x in labels if x in {"partial", "none"}) / n
        avg_rel = sum(rel) / len(rel) if rel else 0.0
        reasons: list[str] = []
        if util < settings.MIN_UTILITY:
            reasons.append("low_utility")
        if avg_rel < settings.MIN_AVG_REL:
            reasons.append("low_relevance")
        if none > settings.MAX_NO_SUPPORT_RATIO:
            reasons.append("high_no_support_ratio")
        if partial_or_no > settings.MAX_PARTIAL_OR_NO_RATIO:
            reasons.append("high_partial_or_no_support_ratio")
        if (
            state.route.risk_level == "high"
            and full < settings.MIN_FULL_SUPPORT_RATIO_HIGH_RISK
        ):
            reasons.append("high_risk_low_full_support")
        if (
            any(x.lower() == "yes" for x in retrieve_decisions)
            and state.loop_count < state.max_loops
        ):
            reasons.append("needs_additional_retrieval")
        reasons.extend([str(x) for x in (parsed.get("insufficiency_hints") or [])])
        reasons = list(dict.fromkeys(reasons))
        su = [
            {
                "full": 1.0 if x == "fully" else 0.0,
                "partial": 1.0 if x == "partial" else 0.0,
                "none": 1.0 if x == "none" else 0.0,
            }
            for x in labels
        ]
        w = (
            settings.SELF_RAG_LOOP0_WEIGHTS
            if state.strictness_level <= 0
            else (
                settings.SELF_RAG_LOOP1_WEIGHTS
                if state.strictness_level == 1
                else settings.SELF_RAG_LOOP2_WEIGHTS
            )
        )
        obj = w[0] * (util / 5.0) + w[1] * full + w[2] * avg_rel
        score = SelfRagScores(
            retrieve_decisions=retrieve_decisions,
            rel_scores=rel,
            support_scores=su,
            utility_score=util,
            avg_isrel=avg_rel,
            no_support_ratio=none,
            partial_or_no_support_ratio=partial_or_no,
            full_support_ratio=full,
            objective_score=obj,
            insufficiency_reasons=reasons,
        )
        c = list(state.answer_candidates)
        c.append(
            {
                "loop_count": state.loop_count,
                "answer": answer,
                "objective_score": obj,
                "scores": score.model_dump(),
            }
        )
        return {"selfrag_scores": score, "answer_candidates": c}

    async def _check_sufficiency_node(self, state: GraphState) -> dict[str, Any]:
        insuf = len(state.selfrag_scores.insufficiency_reasons) > 0
        if insuf and state.loop_count < state.max_loops:
            return {"is_sufficient": False, "next_action": "reinforce"}
        return {"is_sufficient": not insuf, "next_action": "finalize"}

    async def _reinforce_retrieve_node(self, state: GraphState) -> dict[str, Any]:
        rq = state.query
        try:
            p = self._parse_json(
                await self._invoke(
                    self.selfrag_rewrite_prompt,
                    {
                        "query": state.query,
                        "reasons": json.dumps(
                            state.selfrag_scores.insufficiency_reasons,
                            ensure_ascii=False,
                        ),
                        "retrieval_plan": json.dumps(
                            state.global_message_pool.retrieval_plan, ensure_ascii=False
                        ),
                    },
                )
            )
            rq = p.get("rewritten_query") or rq
        except Exception:
            pass
        docs = list(
            (
                await elasticsearch_store.hybrid_search(
                    query=rq, k=max(settings.TOP_K_RESULTS + 1, 3), vector_weight=0.5
                )
            ).documents
        )
        summaries = [
            {
                "doc_index": i,
                "summary": d.content[:180],
                "citation": d.metadata.get("title", d.doc_id),
                "confidence": 0.5,
            }
            for i, d in enumerate(docs)
        ]
        citations = [
            {
                "doc_index": i,
                "citation": s["citation"],
                "title": docs[i].metadata.get("title", "untitled"),
            }
            for i, s in enumerate(summaries)
        ]
        loop = state.loop_count + 1
        strict = min(loop, settings.SELF_RAG_MAX_LOOPS)
        return {
            "loop_count": loop,
            "strictness_level": strict,
            "retrieved_docs": docs,
            "persona_bundle": EvidenceBundleE0(
                top_docs=docs, doc_summaries=summaries, citations_meta=citations
            ),
            "global_message_pool": GlobalMessagePoolM0(
                profile_snapshot=state.global_message_pool.profile_snapshot,
                session_summary=state.global_message_pool.session_summary,
                retrieval_plan={
                    **state.global_message_pool.retrieval_plan,
                    "reinforced_query": rq,
                    "reinforce_reasons": state.selfrag_scores.insufficiency_reasons,
                },
                rerank_notes=state.global_message_pool.rerank_notes,
            ),
            "last_retrieval_query": rq,
        }

    async def _finalize_response_node(self, state: GraphState) -> dict[str, Any]:
        ans = state.draft_answer or state.answer
        if not state.is_sufficient and state.answer_candidates:
            best = max(
                state.answer_candidates,
                key=lambda x: float(x.get("objective_score", 0.0)),
            )
            ans = str(best.get("answer", ans)).strip() or ans
        explain = ""
        try:
            explain = await self._invoke(
                self.transparency_prompt,
                {
                    "query": state.query,
                    "route": json.dumps(state.route.model_dump(), ensure_ascii=False),
                    "loop_count": state.loop_count,
                    "reasons": json.dumps(
                        state.selfrag_scores.insufficiency_reasons, ensure_ascii=False
                    ),
                    "rerank_notes": json.dumps(
                        state.global_message_pool.rerank_notes, ensure_ascii=False
                    ),
                },
            )
        except Exception:
            explain = ""
        if not explain.strip():
            explain = (
                f"Personalized retrieval was applied with policy {state.route.retrieval_policy}. "
                f"Additional retrieval loops: {state.loop_count}. "
                f"Reasons: {', '.join(state.selfrag_scores.insufficiency_reasons) or 'none'}."
            )
        return {
            "final_answer": ans,
            "answer": ans,
            "transparency": {
                "route": state.route.model_dump(),
                "loop_count": state.loop_count,
                "insufficiency_reasons": state.selfrag_scores.insufficiency_reasons,
                "retrieval_plan": state.global_message_pool.retrieval_plan,
                "rerank_notes": state.global_message_pool.rerank_notes,
                "explanation": explain.strip(),
            },
        }

    async def _parse_evidence_response(self, response: str) -> list[int]:
        try:
            m = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            s = (
                m.group(1)
                if m
                else (
                    re.search(r"\{.*?\}", response, re.DOTALL).group(0)
                    if re.search(r"\{.*?\}", response, re.DOTALL)
                    else response
                )
            )
            p = json.loads(s)
            if isinstance(p, dict) and isinstance(p.get("evidence_indices"), list):
                return [int(i) for i in p["evidence_indices"]]
            if isinstance(p, list):
                return [int(i) for i in p]
            return []
        except Exception:
            return (
                [int(n) for n in re.findall(r"\d+", response)]
                if re.findall(r"\d+", response)
                else []
            )

    async def _identify_evidence_node(self, state: GraphState) -> dict[str, Any]:
        docs = state.persona_bundle.top_docs or state.retrieved_docs
        if not docs:
            return {"evidence_indices": []}
        try:
            r = await asyncio.to_thread(
                (self.get_evidence_prompt | self._llm).invoke,
                {
                    "query": state.query,
                    "answer": state.final_answer or state.answer,
                    "documents": self._serialize_docs(docs, 500),
                },
            )
            idx = [
                i
                for i in await self._parse_evidence_response(str(r))
                if 0 <= i < len(docs)
            ]
            kws = [
                (
                    i,
                    self._calculate_keyword_overlap(
                        state.final_answer or state.answer, d.content
                    )[0],
                )
                for i, d in enumerate(docs)
            ]
            for i, s in kws:
                if s >= 0.2:
                    idx.append(i)
            idx = sorted(set(idx))
            if not idx:
                idx = [i for i, s in kws if s >= 0.15] or [0]
            return {"evidence_indices": idx, "retrieved_docs": docs}
        except Exception:
            return {"evidence_indices": list(range(len(docs))), "retrieved_docs": docs}

    def get_graph(self) -> CompiledStateGraph | None:
        return self._graph

    async def prepare_state(
        self, query: str, session_id: str, chat_history: list[Message]
    ) -> GraphState:
        return GraphState(
            query=query,
            chat_history=chat_history,
            session_id=session_id,
            max_loops=settings.SELF_RAG_MAX_LOOPS,
        )

    async def close(self) -> None:
        try:
            await elasticsearch_store.close()
        except Exception:
            pass
        self._initialized = False


rag_graph = RAGGraph()
