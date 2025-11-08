# tests/test_notion_client.py

import pytest
from unittest.mock import MagicMock
from src.clients.notion_client import NotionClient


@pytest.fixture
def mock_requests_get(mocker):
    """'requests.get' 함수를 모방합니다."""
    return mocker.patch("requests.get")


def test_notion_client_init():
    """NotionClient가 토큰으로 헤더를 잘 생성하는지 테스트합니다."""
    client = NotionClient(token="test_token", version="test_ver")
    assert client.headers["Authorization"] == "Bearer test_token"
    assert client.headers["Notion-Version"] == "test_ver"


def test_fetch_all_blocks_success(mock_requests_get):
    """Notion 블록을 재귀적으로 잘 가져오는지 테스트합니다."""

    # 1. Arrange: 가짜 응답 설정
    # 첫 번째 호출 (최상위 블록)
    mock_response_level1 = MagicMock()
    mock_response_level1.status_code = 200
    mock_response_level1.json.return_value = {
        "results": [
            {"id": "block1", "type": "paragraph", "has_children": True},
            {"id": "block2", "type": "paragraph", "has_children": False}
        ]
    }
    # 두 번째 호출 (block1의 하위 블록)
    mock_response_level2 = MagicMock()
    mock_response_level2.status_code = 200
    mock_response_level2.json.return_value = {
        "results": [
            {"id": "block1_child", "type": "paragraph", "has_children": False}
        ]
    }

    # requests.get이 호출 순서대로 다른 응답을 반환하도록 설정
    mock_requests_get.side_effect = [mock_response_level1, mock_response_level2]

    # 2. Act
    client = NotionClient(token="test_token", version="test_ver")
    result = client.fetch_all_blocks("test_page_id")

    # 3. Assert
    assert len(result) == 2
    assert result[0]["id"] == "block1"
    assert "children" in result[0]
    assert result[0]["children"][0]["id"] == "block1_child"
    assert "children" not in result[1]  # block2는 하위 블록 없음


def test_extract_text_content():
    """블록 리스트에서 텍스트가 잘 추출되는지 테스트합니다."""
    sample_blocks = [
        {"type": "child_page", "child_page": {"title": "테스트 제목"}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "첫 번째 문단입니다."}]}}
    ]

    client = NotionClient(token="test_token", version="test_ver")
    text = client.extract_text_content(sample_blocks)

    assert "### 테스트 제목 ###" in text
    assert "첫 번째 문단입니다." in text