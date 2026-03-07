from langchain_core.prompts import ChatPromptTemplate

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_REWRITE_FOR_WEB_SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 사용자 질문을 웹 검색에 더 적합한 형태로 재작성하는 질문 재작성기입니다.
            입력된 질문을 분석하여 그 안에 담긴 의미적 의도와 핵심 의미를 파악하세요.
            그 후 웹 검색 엔진이 관련 정보를 더 잘 찾을 수 있도록 질문을 더 명확하고 구체적인 형태로 재작성하세요.
            
            재작성된 질문만 출력하고 다른 설명은 하지 마세요.""",
        ),
        (
            "user",
            "사용자 질문: {query} \n\n 이 사용자 질문을 웹 검색에 적합하도록 재작성하세요.",
        ),
    ]
)
