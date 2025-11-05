# tests/test_text_processor.py

import pytest
from langchain.schema import Document
import text_processor
from konlpy.tag import Okt  # Okt를 임포트하여 테스트에서 사용


@pytest.fixture
def sample_text():
    """테스트에 사용할 샘플 텍스트를 제공합니다."""
    return """
    ### 첫 번째 문서: 넥스트 커머스 ###
    쿠팡이 잘 나가고, 티메프가 왜 망했는지 알만하다.
    전자상거래는 떠오르는 산업이었지만, 이젠 성숙산업이 되어 버렸지요.

    ### 두 번째 문서: 공산주의자가 온다 ###
    한국형 SF 소설들 모음입니다. 상상력이 기발하여 찬탄하며 읽게 됩니다.
    이신주 작가는 96년생으로 상당히 젊지만, 글 쓴지는 꽤 오래됩니다.
    """


def test_extract_nouns(mocker):
    """extract_nouns가 konlpy.Okt.nouns를 호출하는지 테스트 (모킹)"""

    # 1. 가짜(Mock) Okt 객체 생성
    mock_okt = mocker.MagicMock(spec=Okt)
    mock_okt.nouns.return_value = ["쿠팡", "전자상거래"]

    # 2. _get_okt 함수가 가짜 객체를 반환하도록 설정
    mocker.patch("text_processor._get_okt", return_value=mock_okt)

    # 3. 함수 실행
    nouns = text_processor.extract_nouns("쿠팡과 전자상거래")

    # 4. 검증
    assert nouns == ["쿠팡", "전자상거래"]
    mock_okt.nouns.assert_called_with("쿠팡과 전자상거래")


def test_split_text_with_headers(sample_text, mocker):
    """헤더 기준으로 텍스트가 잘 분리되고 키워드가 추출되는지 테스트합니다."""

    # KeyBERT와 Okt 모두 모킹하여 C++ 및 Java 로드를 피합니다.
    mock_okt = mocker.MagicMock(spec=Okt)
    mock_okt.nouns.return_value = ["쿠팡", "상거래", "산업"]
    mocker.patch("text_processor._get_okt", return_value=mock_okt)

    mock_kw_model = mocker.MagicMock()
    mock_kw_model.extract_keywords.return_value = [("쿠팡", 0.9), ("상거래", 0.8)]
    mocker.patch("text_processor._get_kw_model", return_value=mock_kw_model)

    header_chunks = text_processor.split_text_with_headers(sample_text, top_n=2)

    assert len(header_chunks) == 2
    assert header_chunks[0]["title"] == "첫 번째 문서: 넥스트 커머스"
    assert "쿠팡|상거래" in header_chunks[0]["keywords"]


def test_chunk_text_with_recursive_splitter(sample_text, mocker):
    """재귀적 청커가 잘 작동하는지 테스트합니다."""

    # 이 테스트는 모델 로드가 필요 없으므로 모킹 없이 실행
    # (내부적으로 split_text_with_headers를 호출하므로, 위 테스트의 모킹이 필요할 수 있음)
    # ➡️ 위 테스트와 분리하기 위해 여기서도 모킹을 추가합니다.

    mock_okt = mocker.MagicMock(spec=Okt)
    mock_okt.nouns.return_value = ["쿠팡"]
    mocker.patch("text_processor._get_okt", return_value=mock_okt)

    mock_kw_model = mocker.MagicMock()
    mock_kw_model.extract_keywords.return_value = [("쿠팡", 0.9)]
    mocker.patch("text_processor._get_kw_model", return_value=mock_kw_model)

    final_chunks = text_processor.chunk_text_with_recursive_splitter(sample_text, chunk_size=100, chunk_overlap=20)

    assert len(final_chunks) == 2
    assert isinstance(final_chunks[0], Document)
    assert final_chunks[0].metadata["title"] == "첫 번째 문서: 넥스트 커머스"
    assert "쿠팡이" in final_chunks[0].page_content