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