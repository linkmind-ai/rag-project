"""
search_explorer.py
검색 탐색기 페이지

담당 기능:
  - Vector / Keyword / Hybrid 검색 선택
  - 검색 결과 카드 렌더링 (score, content, metadata)
  - top_k 조정
  - filters (선택적 JSON 입력)
"""

from __future__ import annotations
import json
import streamlit as st

from config import ENDPOINTS, DEFAULT_TOP_K, api_post


# ── Page Renderer ─────────────────────────────────────────────

def render_search_page() -> None:
    st.markdown(
        """
        <div class="page-content">
          <p class="page-title">Search Explorer</p>
          <p class="page-subtitle">인덱스를 직접 쿼리하고 검색 결과를 탐색합니다</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 검색 설정 카드 ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    col_q, col_mode = st.columns([3, 1])
    with col_q:
        query = st.text_input(
            "검색어",
            placeholder="검색할 내용을 입력하세요…",
            label_visibility="collapsed",
        )
    with col_mode:
        mode = st.selectbox(
            "검색 방식",
            ["hybrid", "vector", "keyword"],
            label_visibility="collapsed",
            format_func=lambda x: {"hybrid": "🔀 하이브리드", "vector": "🧬 벡터", "keyword": "🔤 키워드"}[x],
        )

    col_k, col_filter = st.columns([1, 3])
    with col_k:
        top_k = st.slider("Top-K", min_value=1, max_value=20, value=DEFAULT_TOP_K)
    with col_filter:
        filter_raw = st.text_input(
            "Filters (JSON, 선택)",
            placeholder='예) {"source": "notion"}',
            label_visibility="visible",
        )

    search_btn = st.button("검색", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 실행 ──
    if search_btn:
        if not query.strip():
            st.warning("검색어를 입력하세요.")
            return

        filters: dict | None = None
        if filter_raw.strip():
            try:
                filters = json.loads(filter_raw.strip())
            except json.JSONDecodeError:
                st.error("Filters JSON 파싱 오류. 올바른 JSON 형식을 입력하세요.")
                return

        payload = {
            "query": query.strip(),
            "top_k": top_k,
            "filters": filters,
        }

        endpoint_map = {
            "vector":  ENDPOINTS["search_vector"],
            "keyword": ENDPOINTS["search_keyword"],
            "hybrid":  ENDPOINTS["search_hybrid"],
        }

        with st.spinner("검색 중…"):
            result = api_post(endpoint_map[mode], payload, timeout=30)

        if result:
            hits = result.get("results", [])
            total = result.get("total_hits", len(hits))
            proc  = result.get("processing_time", 0)

            # 요약 메트릭
            mode_label = {"hybrid": "하이브리드", "vector": "벡터", "keyword": "키워드"}[mode]
            st.markdown(
                f"""
                <div class="metric-grid">
                  <div class="metric-tile">
                    <div class="metric-value">{total}</div>
                    <div class="metric-label">총 히트</div>
                  </div>
                  <div class="metric-tile">
                    <div class="metric-value">{len(hits)}</div>
                    <div class="metric-label">반환 결과</div>
                  </div>
                  <div class="metric-tile">
                    <div class="metric-value">{proc:.2f}s</div>
                    <div class="metric-label">처리 시간</div>
                  </div>
                  <div class="metric-tile">
                    <div class="metric-value">{mode_label}</div>
                    <div class="metric-label">검색 방식</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not hits:
                st.info("검색 결과가 없습니다.")
                return

            # 결과 카드
            for item in hits:
                score   = item.get("score", 0)
                content = item.get("content", "")
                meta    = item.get("metadata", {})
                doc_id  = item.get("doc_id", "")

                meta_parts = []
                if meta.get("page_title"): meta_parts.append(f"📄 {meta['page_title']}")
                if meta.get("source"):     meta_parts.append(f"src: {meta['source']}")
                if doc_id:                 meta_parts.append(f"id: {doc_id[:16]}…")
                meta_str = " &nbsp;·&nbsp; ".join(meta_parts) if meta_parts else "no metadata"

                preview = content[:320] + ("…" if len(content) > 320 else "")

                st.markdown(
                    f"""
                    <div class="result-card">
                      <span class="result-score">score {score:.4f}</span>
                      <div class="result-content">{_escape(preview)}</div>
                      <div class="result-meta">{meta_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ── Helpers ───────────────────────────────────────────────────

def _escape(text: str) -> str:
    import html
    return html.escape(text)
