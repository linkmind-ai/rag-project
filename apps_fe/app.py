"""
app.py
Notion RAG — Streamlit 메인 진입점

실행:
  streamlit run app.py

파일 구조:
  app.py                ← 메인 진입점 (이 파일)
  config.py             ← API URL, HTTP 헬퍼
  styles.py             ← 전역 CSS
  chat_flow.py          ← 멀티턴 채팅 플로우
  notion_manager.py     ← Notion 페이지 임포트
  document_manager.py   ← 문서 CRUD 관리
  search_explorer.py    ← 검색 탐색기
"""

import streamlit as st

from styles import apply_global_styles
from chat_flow import render_chat_page
from notion_manager import render_notion_page
from document_manager import render_document_page
from search_explorer import render_search_page


# ── Streamlit Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="Notion RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",   # 사이드바 기본 열림 — 좌측 [<] 버튼으로 접기/펼치기
)

# ── Apply Styles ──────────────────────────────────────────────
apply_global_styles()


# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    # Brand Header
    st.markdown(
        """
        <div class="brand-header">
          <div class="brand-icon">✦</div>
          <div>
            <div class="brand-name">Notion RAG</div>
            <div class="brand-sub">Knowledge Assistant</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-section-label">메뉴</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=["💬 채팅", "🗂 Notion 임포트", "📁 문서 관리", "🔍 검색 탐색기"],
        label_visibility="collapsed",
    )

    # ── 하단 상태 표시 ──
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">시스템</div>', unsafe_allow_html=True)

    from config import api_get, ENDPOINTS
    try:
        count_data = api_get(ENDPOINTS["doc_count"])
        if count_data is not None:
            n = count_data.get("count", 0)
            idx = count_data.get("index", "—")
            st.markdown(
                f"""
                <div style="font-size:0.75rem;line-height:1.9">
                  <span class="pill success" style="font-size:0.65rem">● 연결됨</span><br>
                  <span style="color:var(--text-secondary);font-size:0.72rem;font-weight:500">index: {idx}</span><br>
                  <span style="color:var(--text-secondary);font-size:0.72rem;font-weight:500">docs: {n:,}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="pill error" style="font-size:0.65rem">● 서버 미연결</span>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<span class="pill error" style="font-size:0.65rem">● 서버 미연결</span>',
            unsafe_allow_html=True,
        )


# ── Page Router ───────────────────────────────────────────────
if page == "💬 채팅":
    render_chat_page()
elif page == "🗂 Notion 임포트":
    render_notion_page()
elif page == "📁 문서 관리":
    render_document_page()
elif page == "🔍 검색 탐색기":
    render_search_page()