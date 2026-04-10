"""
config.py
API 기본 설정 및 공통 HTTP 클라이언트 유틸리티
"""

import requests
import streamlit as st

# ── API Base URL ──────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

ENDPOINTS = {
    # query
    "query":        f"{API_BASE_URL}/query",
    "query_stream": f"{API_BASE_URL}/query/stream",
    # search
    "search_vector":  f"{API_BASE_URL}/search/vector",
    "search_keyword": f"{API_BASE_URL}/search/keyword",
    "search_hybrid":  f"{API_BASE_URL}/search/hybrid",
    # document
    "doc_add":       f"{API_BASE_URL}/document/add",
    "doc_add_batch": f"{API_BASE_URL}/document/add_batch",
    "doc_upload":    f"{API_BASE_URL}/document/upload",
    "doc_count":     f"{API_BASE_URL}/document/count",
    "doc_delete":    f"{API_BASE_URL}/document",   # DELETE /{doc_id}
    # notion
    "notion_import": f"{API_BASE_URL}/notion/import",
}

DEFAULT_TOP_K = 5
DEFAULT_SESSION_ID = "streamlit-session"


# ── HTTP helpers ──────────────────────────────────────────────

def api_post(url: str, payload: dict, timeout: int = 30) -> dict | None:
    """POST 요청 공통 헬퍼. 에러 발생 시 st.error 표시 후 None 반환."""
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"⚠️ API 오류: {detail}")
    except Exception as e:
        st.error(f"⚠️ 예기치 않은 오류: {e}")
    return None


def api_get(url: str, timeout: int = 10) -> dict | None:
    """GET 요청 공통 헬퍼."""
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API 서버에 연결할 수 없습니다.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"⚠️ API 오류: {detail}")
    except Exception as e:
        st.error(f"⚠️ 예기치 않은 오류: {e}")
    return None


def api_delete(url: str, timeout: int = 10) -> dict | None:
    """DELETE 요청 공통 헬퍼."""
    try:
        res = requests.delete(url, timeout=timeout)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API 서버에 연결할 수 없습니다.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"⚠️ API 오류: {detail}")
    except Exception as e:
        st.error(f"⚠️ 예기치 않은 오류: {e}")
    return None


def api_upload(url: str, file_bytes: bytes, filename: str, metadata: str = "{}") -> dict | None:
    """파일 업로드 공통 헬퍼 (multipart/form-data)."""
    try:
        files = {"file": (filename, file_bytes)}
        data = {"metadata": metadata}
        res = requests.post(url, files=files, data=data, timeout=60)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API 서버에 연결할 수 없습니다.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        st.error(f"⚠️ API 오류: {detail}")
    except Exception as e:
        st.error(f"⚠️ 예기치 않은 오류: {e}")
    return None
