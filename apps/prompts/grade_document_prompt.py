from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# 검색된 문서의 관련성 여부를 이진 점수로 평가하는 데이터 모델
class GradeDocuments(BaseModel):
    """검색된 문서와 질문의 관련도를 0~1 사이 점수로 평가"""

    # 문서가 질문과 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    relevance: float = Field(description="문서와 질문의 관련도")


parser = PydanticOutputParser(pydantic_object=GradeDocuments)
format_instructions = parser.get_format_instructions()

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자 질문과 검색된 문서의 관련성을 평가하는 평가자입니다.

            문서에 질문과 관련된 키워드 또는 의미적으로 관련된 내용이 포함되어 있다면
            관련성이 있는 것으로 판단하세요.

            다음 기준에 따라 0과 1 사이의 실수로 점수를 부여하세요:

            - 0에 가까움: 질문과 거의 관련 없음
            - 0.2~0.4: 일부 키워드 또는 약한 의미적 관련성 있음
            - 0.4~0.7: 부분적으로 관련 있으며 답변에 도움 될 수 있음
            - 0.7~1.0: 질문에 직접적으로 관련 있고 핵심 정보 포함

            중요:
            - 0, 0.5, 1 같은 단순한 값만 사용하지 말고 0.23, 0.61, 0.78처럼 다양한 값을 사용하세요.
            - 가능한 한 세밀하게 점수를 표현하세요.
            
            반드시 아래 형식 지침을 따라 출력하세요.
            {format_instructions}
            """,
        ),
        ("user", "검색된 문서:\n\n{document}\n\n사용자 질문: {query}"),
    ]
).partial(format_instructions=format_instructions)
