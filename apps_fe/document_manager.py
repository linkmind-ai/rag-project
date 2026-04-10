"""
document_manager.py
문서 관리 페이지

담당 기능:
  - 문서 수량 조회 (GET /document/count)
  - 파일 업로드 (POST /document/upload)
  - 텍스트 직접 추가 (POST /document/add)
  - 일괄 추가 - JSON 붙여넣기 (POST /document/add_batch)
  - 문서 삭제 (DELETE /document/{doc_id})
"""

from __future__ import annotations
import json
import streamlit as st

from config import ENDPOINTS, api_get, api_post, api_delete, api_upload


# ── Helpers ───────────────────────────────────────────────────

def _fetch_doc_count() -> int | None:
    result = api_get(ENDPOINTS["doc_count"])
    if result:
        return result.get("count")
    return None


# ── Page Renderer ─────────────────────────────────────────────

def render_document_page() -> None:
    st.markdown(
        """
        <div class="page-content">
          <p class="page-title">Document Manager</p>
          <p class="page-subtitle">인덱스에 저장된 문서를 관리합니다</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 문서 수 카운터 ──
    col_cnt, col_refresh = st.columns([4, 1])
    with col_cnt:
        count = _fetch_doc_count()
        count_str = str(count) if count is not None else "—"
        st.markdown(
            f"""
            <div class="metric-grid" style="margin-bottom:1.4rem">
              <div class="metric-tile">
                <div class="metric-value">{count_str}</div>
                <div class="metric-label">총 문서 수</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_refresh:
        if st.button("↻ 새로고침"):
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Tabs ──
    tab_upload, tab_text, tab_batch, tab_delete = st.tabs(
        ["📁 파일 업로드", "✏️ 텍스트 추가", "📋 일괄 추가", "🗑 문서 삭제"]
    )

    # ────────────────────────────────────────
    # Tab 1: 파일 업로드
    # ────────────────────────────────────────
    with tab_upload:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title">파일 업로드</p>'
            '<p class="section-subtitle">PDF, TXT, MD 등 지원 파일을 업로드하세요</p>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "파일 선택",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )

        with st.expander("메타데이터 설정 (선택)"):
            up_source = st.text_input("source", placeholder="예) manual-docs", key="up_source")
            up_author = st.text_input("author", placeholder="예) team", key="up_author")
            extra_meta: dict = {}
            if up_source: extra_meta["source"] = up_source
            if up_author: extra_meta["author"] = up_author

        if st.button("업로드", use_container_width=True, key="btn_upload") and uploaded:
            file_bytes = uploaded.read()
            with st.spinner("업로드 중…"):
                result = api_upload(
                    ENDPOINTS["doc_upload"],
                    file_bytes,
                    uploaded.name,
                    json.dumps(extra_meta),
                )
            if result and result.get("success"):
                st.markdown(
                    f'<span class="pill success">✓ {result["filename"]} 업로드 완료 — {result["chunks_count"]}개 청크</span>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # Tab 2: 텍스트 직접 추가
    # ────────────────────────────────────────
    with tab_text:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title">텍스트 추가</p>'
            '<p class="section-subtitle">단일 문서를 직접 입력합니다</p>',
            unsafe_allow_html=True,
        )

        content = st.text_area(
            "내용",
            height=180,
            placeholder="여기에 문서 내용을 입력하세요…",
            label_visibility="collapsed",
        )

        col_s, col_a = st.columns(2)
        with col_s:
            txt_source = st.text_input("source", key="txt_source", placeholder="예) meeting-notes")
        with col_a:
            txt_author = st.text_input("author", key="txt_author", placeholder="예) Alice")

        if st.button("추가", use_container_width=True, key="btn_text"):
            if not content.strip():
                st.warning("내용을 입력하세요.")
            else:
                meta: dict = {}
                if txt_source: meta["source"] = txt_source
                if txt_author: meta["author"] = txt_author
                payload = {"content": content.strip(), "metadata": meta}
                with st.spinner("문서 추가 중…"):
                    result = api_post(ENDPOINTS["doc_add"], payload)
                if result and result.get("success"):
                    ids = result.get("document_ids", [])
                    st.markdown(
                        f'<span class="pill success">✓ 추가 완료 — ID: {", ".join(ids)}</span>',
                        unsafe_allow_html=True,
                    )

        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # Tab 3: 일괄 추가 (JSON)
    # ────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title">일괄 추가</p>'
            '<p class="section-subtitle">JSON 배열로 다수 문서를 한 번에 추가합니다</p>',
            unsafe_allow_html=True,
        )

        placeholder_json = '''[
  {"content": "첫 번째 문서 내용", "metadata": {"source": "batch-import"}},
  {"content": "두 번째 문서 내용", "metadata": {"source": "batch-import"}}
]'''
        batch_input = st.text_area(
            "JSON 배열",
            height=220,
            placeholder=placeholder_json,
            label_visibility="collapsed",
        )

        if st.button("일괄 추가", use_container_width=True, key="btn_batch"):
            if not batch_input.strip():
                st.warning("JSON 배열을 입력하세요.")
            else:
                try:
                    docs = json.loads(batch_input.strip())
                    if not isinstance(docs, list):
                        st.error("최상위 구조는 JSON 배열([])이어야 합니다.")
                    else:
                        with st.spinner(f"{len(docs)}개 문서 추가 중…"):
                            result = api_post(ENDPOINTS["doc_add_batch"], docs)
                        if result and result.get("success"):
                            st.markdown(
                                f'<span class="pill success">✓ {result["message"]}</span>',
                                unsafe_allow_html=True,
                            )
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ────────────────────────────────────────
    # Tab 4: 문서 삭제
    # ────────────────────────────────────────
    with tab_delete:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-title">문서 삭제</p>'
            '<p class="section-subtitle">문서 ID를 입력하여 삭제합니다</p>',
            unsafe_allow_html=True,
        )

        doc_id_input = st.text_input(
            "Document ID",
            placeholder="삭제할 문서 ID를 입력하세요",
            label_visibility="collapsed",
        )

        col_del, col_warn = st.columns([1, 3])
        with col_del:
            delete_btn = st.button("삭제", use_container_width=True, key="btn_delete")
        with col_warn:
            st.markdown(
                '<span class="pill warning">⚠ 삭제는 되돌릴 수 없습니다</span>',
                unsafe_allow_html=True,
            )

        if delete_btn:
            if not doc_id_input.strip():
                st.warning("문서 ID를 입력하세요.")
            else:
                url = f"{ENDPOINTS['doc_delete']}/{doc_id_input.strip()}"
                with st.spinner("삭제 중…"):
                    result = api_delete(url)
                if result and result.get("success"):
                    st.markdown(
                        f'<span class="pill success">✓ {result["message"]}</span>',
                        unsafe_allow_html=True,
                    )

        st.markdown("</div>", unsafe_allow_html=True)
