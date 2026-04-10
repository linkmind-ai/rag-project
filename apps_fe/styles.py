"""
styles.py
전역 CSS 스타일 정의
Notion 연동 RAG 시스템 — 지적이고 깔끔한 다크 테마
"""


def apply_global_styles() -> None:
    import streamlit as st

    st.markdown(
        """
<style>
/* ── Google Fonts ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ────────────────────────── */
:root {
  --bg-primary:    #f0f2f8;
  --bg-secondary:  #e8eaf4;
  --bg-card:       #ffffff;
  --bg-input:      #ffffff;
  --accent:        #6a7fc1;
  --accent-soft:   #8fa3d8;
  --accent-glow:   rgba(106, 127, 193, 0.14);
  --text-primary:  #1a1e2e;
  --text-secondary:#3a3f58;
  --text-muted:    #7a80a0;
  --border:        rgba(106, 127, 193, 0.22);
  --border-hover:  rgba(106, 127, 193, 0.55);
  --success:       #3a9e72;
  --warning:       #c8882a;
  --error:         #c94f4f;
  --notion-red:    #e16259;
  --radius-sm:     6px;
  --radius-md:     12px;
  --radius-lg:     18px;
  --shadow:        0 2px 16px rgba(106,127,193,0.10);
  --shadow-lg:     0 6px 32px rgba(106,127,193,0.18);
  --font-display:  'DM Serif Display', serif;
  --font-body:     'DM Sans', sans-serif;
  --font-mono:     'JetBrains Mono', monospace;
}

/* ── Global Reset ─────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App Layout ───────────────────────────── */
.main .block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #e9ecf8 0%, #dfe4f5 100%) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 2px 0 12px rgba(106,127,193,0.08);
}
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1.2rem !important;
}

/* ── Sidebar collapse button ──────────────── */
[data-testid="stSidebarCollapseButton"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent) !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stSidebarCollapseButton"]:hover {
  background: var(--accent-glow) !important;
  border-color: var(--border-hover) !important;
}
/* Expand button (when collapsed) */
[data-testid="stSidebarOpenButton"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--accent) !important;
  box-shadow: var(--shadow) !important;
}
[data-testid="stSidebarOpenButton"]:hover {
  background: var(--accent-glow) !important;
}

/* ── Sidebar Logo / Brand ─────────────────── */
.brand-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0.4rem 0 1.6rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.6rem;
}
.brand-icon {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  color: #fff;
  box-shadow: 0 2px 8px rgba(106,127,193,0.25);
}
.brand-name {
  font-family: var(--font-display) !important;
  font-size: 1.15rem !important;
  color: var(--text-primary) !important;
  letter-spacing: 0.01em;
  line-height: 1.2;
  font-weight: 700;
}
.brand-sub {
  font-size: 0.68rem;
  color: var(--text-muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-weight: 500;
}

/* ── Nav Labels ───────────────────────────── */
.nav-section-label {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 1.4rem 0 0.5rem;
}

/* ── Streamlit Radio as nav ───────────────── */
[data-testid="stSidebar"] .stRadio > label {
  display: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  gap: 2px !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.55rem 0.75rem !important;
  cursor: pointer !important;
  transition: background 0.15s ease !important;
  width: 100%;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {
  background: rgba(106,127,193,0.12) !important;
}

/* ── Buttons ──────────────────────────────── */
.stButton > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  padding: 0.45rem 1.1rem !important;
  transition: all 0.2s ease !important;
  letter-spacing: 0.02em;
  box-shadow: 0 2px 8px rgba(106,127,193,0.2);
}
.stButton > button:hover {
  background: #5a6fb1 !important;
  box-shadow: 0 4px 16px rgba(106,127,193,0.3) !important;
  transform: translateY(-1px) !important;
}

/* ── Inputs ───────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
  background: var(--bg-input) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stTextInput > label, .stTextArea > label {
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
}

/* ── Chat container wrapper ───────────────── */
.chat-wrapper {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 0px);
  background: var(--bg-primary);
}

.chat-header {
  padding: 1.2rem 2rem 1rem;
  border-bottom: 1px solid var(--border);
  background: linear-gradient(90deg, #eef0fa 0%, #f0f2f8 100%);
  flex-shrink: 0;
}
.chat-header-title {
  font-family: var(--font-display) !important;
  font-size: 1.35rem;
  color: var(--text-primary);
  margin: 0;
  font-weight: 700;
}
.chat-header-sub {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 2px;
  letter-spacing: 0.04em;
  font-weight: 500;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1.6rem 2rem 1rem;
  scroll-behavior: smooth;
}

/* ── Message bubbles ──────────────────────── */
.msg-row {
  display: flex;
  margin-bottom: 1.4rem;
  gap: 10px;
  animation: fadeUp 0.3s ease forwards;
}
.msg-row.user   { flex-direction: row-reverse; }
.msg-row.assistant { flex-direction: row; }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

.msg-avatar {
  width: 30px; height: 30px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
  flex-shrink: 0;
  margin-top: 2px;
}
.msg-avatar.user {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
  color: #fff;
}
.msg-avatar.assistant {
  background: #fff;
  border: 1.5px solid var(--border);
  color: var(--accent);
}

.msg-bubble {
  max-width: 72%;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: 0.875rem;
  line-height: 1.65;
  font-weight: 450;
}
.msg-bubble.user {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%);
  color: #fff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 2px 10px rgba(106,127,193,0.22);
}
.msg-bubble.assistant {
  background: #ffffff;
  color: var(--text-primary);
  border: 1.5px solid var(--border);
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(106,127,193,0.08);
}

/* Sources inside assistant bubble */
.sources-toggle {
  margin-top: 0.65rem;
  padding-top: 0.6rem;
  border-top: 1px solid var(--border);
  font-size: 0.72rem;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  font-weight: 600;
}
.source-chip {
  display: inline-block;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 7px;
  margin: 3px 2px 0;
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--accent);
  font-weight: 600;
}

/* ── Chat input bar (pinned bottom) ───────── */
.chat-input-bar {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: #eef0fa;
  padding: 0.85rem 2rem;
}

/* Streamlit chat_input overrides */
[data-testid="stChatInput"] {
  border-top: none !important;
  background: transparent !important;
  padding: 0 !important;
}
[data-testid="stChatInputTextArea"] {
  background: #ffffff !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-family: var(--font-body) !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  resize: none !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stChatInputTextArea"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── Section Cards ────────────────────────── */
.section-card {
  background: var(--bg-card);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.6rem;
  margin-bottom: 1.2rem;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
  box-shadow: var(--shadow);
}
.section-card:hover {
  border-color: var(--border-hover);
  box-shadow: var(--shadow-lg);
}

.section-title {
  font-family: var(--font-display) !important;
  font-size: 1.05rem;
  color: var(--text-primary);
  margin-bottom: 0.15rem;
  font-weight: 700;
}
.section-subtitle {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-bottom: 1.1rem;
  letter-spacing: 0.03em;
  font-weight: 500;
}

/* ── Status Pills ─────────────────────────── */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.pill.success { background: rgba(58,158,114,.12); color: var(--success); border: 1px solid rgba(58,158,114,.3); }
.pill.warning { background: rgba(200,136,42,.10); color: var(--warning); border: 1px solid rgba(200,136,42,.3); }
.pill.error   { background: rgba(201,79,79,.10);  color: var(--error);   border: 1px solid rgba(201,79,79,.3);  }
.pill.info    { background: var(--accent-glow);   color: var(--accent);  border: 1px solid var(--border); }

/* ── Metric Tiles ─────────────────────────── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 0.8rem;
  margin-bottom: 1.2rem;
}
.metric-tile {
  background: linear-gradient(135deg, #eef0fa 0%, #f5f6fb 100%);
  border: 1.5px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.9rem 1rem;
  text-align: center;
}
.metric-value {
  font-family: var(--font-mono);
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--accent);
}
.metric-label {
  font-size: 0.68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 2px;
  font-weight: 600;
}

/* ── Search Result Cards ──────────────────── */
.result-card {
  background: #ffffff;
  border: 1.5px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-md);
  padding: 0.9rem 1rem;
  margin-bottom: 0.7rem;
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
  box-shadow: var(--shadow);
}
.result-card:hover {
  transform: translateX(2px);
  border-left-color: var(--accent-soft);
  box-shadow: var(--shadow-lg);
}
.result-score {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--accent);
  float: right;
  margin-top: 1px;
  font-weight: 600;
  background: var(--accent-glow);
  padding: 1px 6px;
  border-radius: 4px;
}
.result-content {
  font-size: 0.82rem;
  color: var(--text-secondary);
  line-height: 1.6;
  margin-top: 0.3rem;
  font-weight: 450;
}
.result-meta {
  font-size: 0.68rem;
  color: var(--text-muted);
  font-family: var(--font-mono);
  margin-top: 0.5rem;
  font-weight: 500;
}

/* ── Expander ─────────────────────────────── */
[data-testid="stExpander"] {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background: #fafbff !important;
}
[data-testid="stExpander"] summary {
  color: var(--text-secondary) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}

/* ── Divider ──────────────────────────────── */
hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.2rem 0;
}

/* ── Scrollbar ────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(106,127,193,0.25); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Spinner ──────────────────────────────── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Misc ─────────────────────────────────── */
.stMarkdown p { color: var(--text-secondary); font-size: 0.88rem; line-height: 1.65; font-weight: 450; }
.stAlert { border-radius: var(--radius-md) !important; font-size: 0.83rem !important; }
code { font-family: var(--font-mono) !important; font-size: 0.82em !important; color: var(--accent) !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
  color: var(--text-muted) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
  background: var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
  background: var(--border) !important;
}

/* Toggle */
[data-testid="stToggle"] label {
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label {
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  font-size: 0.8rem !important;
}

/* Page content wrapper */
.page-content {
  padding: 1.8rem 2rem;
  max-width: 860px;
}
.page-title {
  font-family: var(--font-display) !important;
  font-size: 1.6rem;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
  font-weight: 700;
}
.page-subtitle {
  font-size: 0.78rem;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  margin-bottom: 2rem;
  font-weight: 500;
}

/* Checkbox, slider labels */
[data-testid="stCheckbox"] label span,
[data-testid="stSlider"] label {
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 0.83rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: #fafbff !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploader"] span {
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
}

/* Warning / info / error boxes */
[data-testid="stAlert"] {
  color: var(--text-primary) !important;
}
</style>
""",
        unsafe_allow_html=True,
    )