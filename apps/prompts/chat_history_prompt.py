from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_CHAT_WITH_HISTORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 주어진 질문과 답변으로 사용자에게 답변을 하는 AI 어시스턴트입니다.
            제공된 컨텍스트 문서와 대화 이력을 바탕으로 질문에 답변해주세요.

            컨텍스트:
            {context}
            
            답변시 다음을 지켜주세요:
            1. 반드시 위 컨텍스트에 포함된 내용을 우선적으로 사용하여 답변하세요.
            2. 대화 이력을 참고하여 일관된 답변을 제공하세요.
            3. 컨텍스트에서 답을 찾을 수 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
            4. 컨텍스트의 언어에 상관없이, 질문에 대한 답변은 반드시 한국어로 제공하세요.
            5. '컨텍스트에 따르면~', '컨텍스트 문서에는~' 등의 표현으로 답변을 시작하지 마세요.""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{query}"),
    ]
)
