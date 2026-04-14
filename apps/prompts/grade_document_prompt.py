from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated


# 검색된 문서의 관련성 여부를 이진 점수로 평가하는 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서와 질문의 관련도를 0~1 사이 점수로 평가"""

    relevance: Annotated[float, Field(ge=0.0, le=1.0)]


# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자 질문과 검색된 문서의 관련성을 평가하는 평가자입니다.

            다음 기준에 따라 0.0과 1.0 사이의 실수(float)로 점수를 부여하세요:

            - 0.0에 가까움: 질문과 거의 관련 없음
            - 0.2~0.4: 일부 키워드 또는 약한 의미적 관련성 있음
            - 0.4~0.7: 부분적으로 관련 있으며 답변에 도움 될 수 있음
            - 0.7~1.0: 질문에 직접적으로 관련 있고 핵심 정보 포함

            중요:
            - 반드시 0.0 이상 1.0 이하의 소수(float)만 출력하세요.
            - 절대로 1보다 큰 값이나 음수를 출력하지 마세요.
            - 절대로 null, None, 빈 값을 출력하지 마세요.

            출력 형식:
            - 반드시 JSON 객체 하나만 출력하세요.
            - 반드시 "relevance" 필드만 포함하세요.
            - 다른 텍스트는 절대 출력하지 마세요.

            출력 예:
            {{"relevance": 0.73}}
            """,
        ),
        ("user", "검색된 문서:\n\n{document}\n\n사용자 질문: {query}"),
    ]
)
