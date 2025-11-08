# tests/test_text_processor.py

import pytest
from langchain.schema import Document
from src.common.text_processor import TextProcessor


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


@pytest.fixture
def mock_models(mocker):
    """TextProcessor의 모델 로딩 메서드(_get_okt, _get_kw_model)를 모방합니다."""

    # 1. _get_okt 모방
    mock_okt = mocker.MagicMock()
    mock_okt.nouns.return_value = ["쿠팡", "상거래", "소설", "상상력"]
    mocker.patch(
        "src.common.text_processor.TextProcessor._get_okt",
        return_value=mock_okt
    )

    # 2. _get_kw_model 모방
    mock_kw_model = mocker.MagicMock()
    mock_kw_model.extract_keywords.return_value = [("키워드1", 0.9), ("키워드2", 0.8)]
    mocker.patch(
        "src.common.text_processor.TextProcessor._get_kw_model",
        return_value=mock_kw_model
    )

    return mock_okt, mock_kw_model


def test_chunk_text_with_headers_and_keywords(sample_text, mock_models):
    """
    TextProcessor.chunk_text가 헤더 분리 및 키워드 추출을 잘 수행하는지 테스트
    """
    mock_okt, mock_kw_model = mock_models

    # 1. Act
    # sbert_model_name은 중요하지 않음 (어차피 _get_kw_model이 모방됨)
    processor = TextProcessor(sbert_model_name="fake-model")
    final_chunks = processor.chunk_text(sample_text, chunk_size=150, chunk_overlap=20)

    # 2. Assert
    # 100자보다 본문이 길기 때문에 2개 헤더가 총 4개의 청크로 나뉘어야 함
    assert len(final_chunks) == 4

    # 첫 번째 청크가 올바른 메타데이터를 가졌는지 확인
    assert isinstance(final_chunks[0], Document)
    assert final_chunks[0].metadata["title"] == "첫 번째 문서: 넥스트 커머스"
    assert final_chunks[0].metadata["keywords"] == "키워드1|키워드2"
    assert "쿠팡이 잘 나가고" in final_chunks[0].page_content

    # 네 번째 청크가 올바른 메타데이터를 가졌는지 확인
    assert final_chunks[3].metadata["title"] == "두 번째 문서: 공산주의자가 온다"
    assert "글 쓴지는 꽤 오래됩니다" in final_chunks[3].page_content

    # 내부 함수들이 호출되었는지 확인
    assert mock_okt.nouns.call_count == 2
    assert mock_kw_model.extract_keywords.call_count == 2