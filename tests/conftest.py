"""
pytest 설정 파일

pytest-asyncio 모드 및 공통 fixture 정의
"""

import sys
from pathlib import Path

import pytest

# apps 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

# pytest-asyncio 모드 설정
pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config: pytest.Config) -> None:
    """pytest 설정 커스터마이징"""
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(scope="session")
def groq_eval_llm():
    """RAGAS 평가 judge용 Groq LLM — KEY_2 우선, 없으면 KEY_1 사용 (session scope)"""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        pytest.skip("langchain-groq 미설치 — .venv-eval 환경에서 실행하세요.")

    from common.config import settings

    key = settings.GROQ_API_KEY_2 or settings.GROQ_API_KEY
    if not key:
        pytest.skip("GROQ_API_KEY 미설정 — .env 확인")
    return ChatGroq(api_key=key, model="llama-3.3-70b-versatile")