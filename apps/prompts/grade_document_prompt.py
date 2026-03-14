from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자 질문과 검색된 문서의 관련성을 평가하는 평가자입니다.

            문서에 질문과 관련된 키워드 또는 의미적으로 관련된 내용이 포함되어 있다면
            관련성이 있는 것으로 판단하세요.

            반드시 아래 JSON 형식으로만 답하세요.

            {{"binary_score": "yes"}} 또는 {{"binary_score": "no"}}

            추가 설명은 절대 하지 마세요.
            """,
        ),
        ("user", "검색된 문서:\n\n{document}\n\n사용자 질문: {query}"),
    ]
)


# 검색된 문서의 관련성 여부를 이진 점수로 평가하는 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성을 'yes' 또는 'no'로 판단하기 위한 이진 점수 모델."""

    # 문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: Literal["yes", "no"] = Field(
        description="문서가 질문과 관련이 있는지 여부 ('yes' 또는 'no')."
    )
