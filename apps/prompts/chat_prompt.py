from langchain_core.prompts import ChatPromptTemplate

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 주어진 컨텍스트를 바탕으로 사용자의 질문에 답변하는 AI 어시스턴트입니다.

            컨텍스트:
            {context}

            답변 규칙:
            1. 반드시 위 컨텍스트에 포함된 내용만을 사용하여 답변하세요.
            2. 컨텍스트에 없는 내용은 절대 추가하지 마세요. 
            3. 컨텍스트에서 확인된 핵심 사실을 중심으로 답변하세요.
            4. 답변은 3~4문장으로 답하세요.
            5. 컨텍스트에서 답변을 찾을 수 없는 경우, 모른다고 솔직하게 답하세요.
            6. 답변은 한국어로 제공하세요.""",
        ),
        ("user", "{query}"),
    ]
)
