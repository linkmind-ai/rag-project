"""
notion_manager.py
노션 페이지 가져오기 (POST /notion/import) 관리 페이지

담당 기능:
  - 노션 페이지 ID 입력 및 임포트
  - recursive 옵션 (하위 페이지 포함)
  - 임포트 결과 표시 (페이지 제목, 청크 수, 문서 ID)
  - 임포트 히스토리 (세션 내)
"""

from __future__ import annotations
import streamlit as st
from config import ENDPOINTS, api_post


# ── Session State ─────────────────────────────────────────────

def _init_state() -> None:
    if "notion_history" not in st.session_state:
        st.session_state.notion_history: list[dict] = []


# ── Page Renderer ─────────────────────────────────────────────

def render_notion_page() -> None:
    _init_state()

    st.markdown(
        """
        <div class="page-content">
          <p class="page-title">Notion Import</p>
          <p class="page-subtitle">NOTION 페이지를 가져와 RAG 인덱스에 임베딩합니다</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Import Card ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">페이지 가져오기</p>'
        '<p class="section-subtitle">Notion 페이지 URL 또는 ID를 입력하세요</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        page_input = st.text_input(
            "Page ID / URL",
            placeholder="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  또는  https://www.notion.so/...",
            label_visibility="collapsed",
        )
    with col2:
        recursive = st.checkbox("하위 페이지 포함", value=False)

    # Metadata (선택)
    with st.expander("추가 메타데이터 (선택)"):
        meta_source  = st.text_input("source",  placeholder="예) personal-notes")
        meta_author  = st.text_input("author",  placeholder="예) John Doe")
        meta_tags    = st.text_input("tags",    placeholder="예) project, idea")

    import_btn = st.button("✦ 가져오기", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Action ──
    if import_btn:
        if not page_input.strip():
            st.warning("페이지 ID 또는 URL을 입력하세요.")
        else:
            page_id = _extract_page_id(page_input.strip())
            metadata: dict = {}
            if meta_source: metadata["source"] = meta_source
            if meta_author: metadata["author"] = meta_author
            if meta_tags:   metadata["tags"]   = [t.strip() for t in meta_tags.split(",")]

            payload = {
                "page_id":   page_id,
                "recursive": recursive,
                "metadata":  metadata,
            }

            with st.spinner("노션 페이지 가져오는 중…"):
                result = api_post(ENDPOINTS["notion_import"], payload, timeout=60)

            if result and result.get("success"):
                st.markdown(
                    f'<span class="pill success">✓ 임포트 성공</span>',
                    unsafe_allow_html=True,
                )
                st.session_state.notion_history.append(
                    {
                        "page_id":    result.get("page_id", page_id),
                        "title":      result.get("page_title", "Untitled"),
                        "chunks":     result.get("chunks_count", 0),
                        "doc_ids":    result.get("document_ids", []),
                    }
                )

                # 결과 요약
                st.markdown(
                    f"""
                    <div class="section-card" style="margin-top:1rem">
                      <p class="section-title">{result.get("page_title", "Untitled")}</p>
                      <div class="metric-grid">
                        <div class="metric-tile">
                          <div class="metric-value">{result.get("chunks_count", 0)}</div>
                          <div class="metric-label">청크</div>
                        </div>
                        <div class="metric-tile">
                          <div class="metric-value">{len(result.get("document_ids", []))}</div>
                          <div class="metric-label">문서 ID</div>
                        </div>
                      </div>
                      <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">
                        page_id: {result.get("page_id", page_id)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Import History ──
    if st.session_state.notion_history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title" style="margin-bottom:0.8rem">임포트 히스토리</p>',
            unsafe_allow_html=True,
        )
        for item in reversed(st.session_state.notion_history):
            st.markdown(
                f"""
                <div class="result-card">
                  <span class="result-score">{item['chunks']} chunks</span>
                  <strong style="font-size:0.85rem;color:var(--text-primary)">{item['title']}</strong>
                  <div class="result-meta">page_id: {item['page_id']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Helpers ───────────────────────────────────────────────────

def _extract_page_id(raw: str) -> str:
    """URL 또는 순수 ID에서 32자리 page_id를 추출."""
    # URL 형태: https://www.notion.so/Title-{id} or .../p/{id}
    if "notion.so" in raw:
        parts = raw.rstrip("/").split("/")
        candidate = parts[-1]
        # URL 파라미터 제거
        candidate = candidate.split("?")[0].split("#")[0]
        # 마지막 '-' 뒤의 32자리
        if "-" in candidate:
            candidate = candidate.rsplit("-", 1)[-1]
        return candidate.replace("-", "")
    # 이미 ID 형태
    return raw.replace("-", "")
