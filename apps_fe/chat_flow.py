"""
chat_flow.py
채팅 플로우 관리 — 멀티턴 질의응답 인터페이스

담당 기능:
  - 세션 히스토리 관리 (st.session_state)
  - /query/stream SSE 스트리밍 처리
  - 메시지 버블 렌더링
  - 소스 출처 표시
"""

from __future__ import annotations

import json
import uuid
import time

import requests
import streamlit as st

from config import ENDPOINTS, DEFAULT_SESSION_ID, api_post


# ── Session State Initializer ─────────────────────────────────

def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list[dict]
    if "session_id" not in st.session_state:
        st.session_state.session_id = DEFAULT_SESSION_ID
    if "use_history" not in st.session_state:
        st.session_state.use_history = True
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True


# ── Message Renderers ─────────────────────────────────────────

def _render_message(role: str, content: str, sources: list[dict] | None = None) -> None:
    """단일 메시지 버블을 HTML로 렌더링."""
    avatar = "👤" if role == "user" else "✦"
    bubble_class = role  # "user" | "assistant"

    sources_html = ""
    if sources:
        chips = "".join(
            f'<span class="source-chip">#{s["index"]} {_truncate(s.get("metadata", {}).get("page_title", ""), 18)}</span>'
            for s in sources[:5]
        )
        sources_html = f'<div class="sources-toggle">SOURCES &nbsp;{chips}</div>'

    html = f"""
<div class="msg-row {bubble_class}">
  <div class="msg-avatar {bubble_class}">{avatar}</div>
  <div class="msg-bubble {bubble_class}">
    {_md_to_html(content)}
    {sources_html}
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def _truncate(text: str, n: int) -> str:
    return text[:n] + "…" if len(text) > n else text


def _md_to_html(text: str) -> str:
    """마크다운 줄바꿈을 간단히 HTML로 변환."""
    import html as h
    escaped = h.escape(text)
    # 줄바꿈 처리
    escaped = escaped.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p style='margin:0'>{escaped}</p>"


def render_chat_history() -> None:
    """저장된 전체 채팅 히스토리를 렌더링."""
    for msg in st.session_state.messages:
        _render_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
        )


# ── Streaming Query ───────────────────────────────────────────

# 서버가 stream 이벤트에서 사용할 수 있는 텍스트 키 후보
_STREAM_CONTENT_KEYS = ("content", "chunk", "text", "delta", "token")


def _extract_chunk(event: dict) -> str:
    """stream 이벤트에서 텍스트 청크를 추출. 여러 키 이름을 방어적으로 시도."""
    for key in _STREAM_CONTENT_KEYS:
        val = event.get(key)
        if val is not None:
            return str(val)
    return ""


def _stream_query(query: str) -> dict:
    """
    /query/stream SSE 엔드포인트를 호출하여 스트리밍 응답을 처리한다.
    반환: {"answer": str, "sources": list}

    수정사항:
    - finally에서 stream_placeholder를 무조건 지우는 버그 수정
      → 스트리밍이 정상 완료됐을 때만 placeholder를 정리
    - stream 이벤트의 텍스트 키를 방어적으로 탐색
    - sources를 done / generate_end 양쪽에서 모두 수집
    - 수신한 raw 이벤트를 expander로 디버그 출력 (개발 중 확인용)
    """
    payload = {
        "session_id": st.session_state.session_id,
        "query": query,
        "use_history": st.session_state.use_history,
    }

    answer_parts: list[str] = []
    sources: list[dict] = []
    error_occurred = False

    status_placeholder = st.empty()
    stream_placeholder = st.empty()

    # 개발/디버그용 — 수신 이벤트 확인
    debug_events: list[str] = []

    try:
        with requests.post(
            ENDPOINTS["query_stream"],
            json=payload,
            stream=True,
            timeout=120,
            headers={
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
            },
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                # iter_lines()는 빈 줄(keep_blank_lines=False 기본값)을 이미 걸러줌
                # 그래도 명시적으로 체크
                if not raw_line:
                    continue

                line: str = (
                    raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                )

                # SSE 형식: "data: {...}"
                if not line.startswith("data:"):
                    continue

                raw_json = line[5:].strip()
                if not raw_json:
                    continue

                try:
                    event = json.loads(raw_json)
                except json.JSONDecodeError as je:
                    debug_events.append(f"[JSON 파싱 실패] {raw_json[:120]} — {je}")
                    continue

                etype = event.get("type", "")
                debug_events.append(f"[{etype}] {raw_json[:200]}")

                # ── 이벤트별 처리 ──────────────────────────
                if etype == "session_id":
                    pass  # 세션 확인용, 무시

                elif etype == "retrieve_start":
                    status_placeholder.markdown(
                        '<span class="pill info">🔍 관련 문서 검색 중…</span>',
                        unsafe_allow_html=True,
                    )

                elif etype == "retrieve_end":
                    status_placeholder.markdown(
                        '<span class="pill success">📄 문서 검색 완료</span>',
                        unsafe_allow_html=True,
                    )

                elif etype == "generate_start":
                    status_placeholder.markdown(
                        '<span class="pill info">✦ 답변 생성 중…</span>',
                        unsafe_allow_html=True,
                    )

                elif etype in ("stream", "content"):
                    # 실제 서버는 "content" 타입으로 청크를 전송
                    chunk = _extract_chunk(event)
                    if chunk:
                        answer_parts.append(chunk)
                    current = "".join(answer_parts)
                    stream_placeholder.markdown(current + "▌")

                elif etype == "generate_end":
                    if event.get("sources"):
                        sources = event["sources"]
                    status_placeholder.empty()

                elif etype in (
                    "identify_evidence_start", "identify_evidence_end",
                    "evidence_start", "web_search_decision",
                ):
                    pass  # UI 표시 불필요

                elif etype == "evidence_end":
                    # sources는 evidence_end 의 evidence_docs 에 담겨 옴
                    raw_docs = event.get("evidence_docs", [])
                    if raw_docs:
                        sources = raw_docs

                elif etype == "done":
                    # done 에 full_response 가 있으면 answer_parts 가 빌 때 보완
                    if not answer_parts and event.get("full_response"):
                        answer_parts.append(event["full_response"])
                    if event.get("sources"):
                        sources = event["sources"]
                    status_placeholder.empty()

                elif etype == "error":
                    error_msg = event.get("error", event.get("message", "알 수 없는 오류"))
                    st.error(f"⚠️ 서버 오류: {error_msg}")
                    error_occurred = True

    except requests.exceptions.ConnectionError:
        st.error("⚠️ API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        error_occurred = True
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"⚠️ API HTTP 오류 ({e.response.status_code if e.response else '?'}): {detail or e}")
        error_occurred = True
    except Exception as e:
        st.error(f"⚠️ 스트리밍 오류: {type(e).__name__}: {e}")
        error_occurred = True
    finally:
        # status는 항상 정리
        status_placeholder.empty()
        # stream placeholder는 answer가 있을 때만 정리
        # (answer가 없으면 에러 상황이므로 이미 비어 있음)
        if answer_parts:
            stream_placeholder.empty()

    # 디버그 패널 (개발 중 활성화 — 운영 시 아래 블록 주석 처리)
    if debug_events:
        with st.expander("🛠 스트림 이벤트 로그 (디버그)", expanded=False):
            st.code("\n".join(debug_events), language="json")

    answer = "".join(answer_parts)
    return {"answer": answer, "sources": sources}


# ── Non-streaming Query ───────────────────────────────────────

def _sync_query(query: str) -> dict:
    """/query 동기 엔드포인트 호출."""
    payload = {
        "session_id": st.session_state.session_id,
        "query": query,
        "use_history": st.session_state.use_history,
    }
    with st.spinner("답변 생성 중…"):
        result = api_post(ENDPOINTS["query"], payload, timeout=60)

    if result:
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
    return {"answer": "", "sources": []}


# ── Main Chat Page ────────────────────────────────────────────

def render_chat_page() -> None:
    init_chat_state()

    # ── 상단 헤더 ──
    st.markdown(
        """
        <div class="chat-header">
          <p class="chat-header-title">✦ Ask your Notes</p>
          <p class="chat-header-sub">Notion 기반 RAG · 멀티턴 질의응답</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 설정 패널 (사이드바에서 제어) ──
    with st.sidebar:
        st.markdown('<div class="nav-section-label">채팅 설정</div>', unsafe_allow_html=True)
        st.session_state.use_history = st.toggle(
            "멀티턴 히스토리", value=st.session_state.use_history
        )
        st.session_state.use_streaming = st.toggle(
            "스트리밍 응답", value=st.session_state.use_streaming
        )
        st.session_state.session_id = st.text_input(
            "세션 ID", value=st.session_state.session_id
        )

        st.markdown('<div class="nav-section-label">세션</div>', unsafe_allow_html=True)
        if st.button("💬 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
            st.rerun()

    # ── 채팅 히스토리 스크롤 영역 ──
    chat_area = st.container()
    with chat_area:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="text-align:center; padding: 4rem 0; color: var(--text-muted);">
                  <div style="font-size:2.5rem; margin-bottom:0.8rem;">✦</div>
                  <div style="font-family:var(--font-display); font-size:1.1rem; color:var(--text-secondary); margin-bottom:0.5rem;">
                    무엇이든 물어보세요
                  </div>
                  <div style="font-size:0.78rem; letter-spacing:0.06em; text-transform:uppercase;">
                    Notion 페이지를 먼저 가져온 뒤 질문하세요
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_chat_history()

    # ── 하단 고정 입력창 ──
    # st.chat_input 은 자동으로 페이지 하단에 고정됨
    user_input = st.chat_input("노트에 대해 질문하세요…")

    if user_input and user_input.strip():
        query = user_input.strip()

        # 사용자 메시지 저장 & 즉시 렌더링
        st.session_state.messages.append({"role": "user", "content": query})
        _render_message("user", query)

        # 응답 생성
        if st.session_state.use_streaming:
            result = _stream_query(query)
        else:
            result = _sync_query(query)

        answer  = result.get("answer", "")
        sources = result.get("sources", [])

        if answer:
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            _render_message("assistant", answer, sources)
            st.rerun()
        else:
            st.warning("응답을 받지 못했습니다. 다시 시도해주세요.")