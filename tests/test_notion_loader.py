# tests/test_notion_loader.py

import pytest
from unittest.mock import MagicMock
import notion_loader


# 'mocker'는 pytest-mock이 제공하는 픽스처입니다.
def test_get_page_content_success(mocker):
    """Notion API 호출이 성공했을 때를 테스트합니다."""

    # 1. 가짜(Mock) 응답 객체 생성
    mock_response_page = MagicMock()
    mock_response_page.status_code = 200
    mock_response_page.json.return_value = {"id": "page123", "properties": {}}

    mock_response_blocks = MagicMock()
    mock_response_blocks.status_code = 200
    mock_response_blocks.json.return_value = {"results": [{"type": "paragraph"}]}

    # 2. 'requests.get'이 가짜 응답을 반환하도록 설정
    mocker.patch(
        "requests.get",
        side_effect=[mock_response_page, mock_response_blocks]  # 첫 호출, 두 번째 호출
    )

    # 3. 함수 실행
    result = notion_loader.get_page_content("test-page-id")

    # 4. 검증
    assert result is not None
    assert result["id"] == "page123"
    assert "content" in result
    assert result["content"][0]["type"] == "paragraph"


def test_get_page_content_fail(mocker):
    """Notion API 호출이 실패했을 때를 테스트합니다."""

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mocker.patch("requests.get", return_value=mock_response)

    result = notion_loader.get_page_content("bad-page-id")

    # 401 에러가 났으므로 None이 반환되어야 함
    assert result is None


def test_extract_text_content():
    """블록 리스트에서 텍스트가 잘 추출되는지 테스트합니다."""
    sample_blocks = [
        {"type": "child_page", "child_page": {"title": "테스트 제목"}},
        {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "첫 번째 문단입니다."}]}}
    ]

    text = notion_loader.extract_text_content(sample_blocks)

    assert "### 테스트 제목 ###" in text
    assert "첫 번째 문단입니다." in text