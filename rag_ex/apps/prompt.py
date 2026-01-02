from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# 프롬프트에 적정 토큰수 적용하는 내용 추가 필요
_CHAT_PROMPT = ChatPromptTemplate.from_messages([
                ("system", """당신은 주어진 질문과 답변으로 사용자에게 답변을 하는 AI 어시스턴트입니다.
                 제공된 컨텍스트 문서와 대화 이력을 바탕으로 질문에 답변해주세요.
                 
                 컨텍스트:
                 {context}
                 
                 답변시 다음을 지켜주세요:
                 1. 컨텍스트에 있는 정보를 우선적으로 활용하세요.
                 2. 확실하지 않은 내용은 모른다고 답변하세요.
                 3. 질문에 대한 답변은 한국어로 제공하세요."""),
                 ("user", "{query}")
            ]).strip()

_CHAT_WITH_HISTORY_PROMPT = ChatPromptTemplate.from_messages([
                ("system", """당신은 주어진 질문과 답변으로 사용자에게 답변을 하는 AI 어시스턴트입니다.
                 제공된 컨텍스트 문서와 대화 이력을 바탕으로 질문에 답변해주세요.
                 
                 컨텍스트:
                 {context}
                 
                 답변시 다음을 지켜주세요:
                 1. 컨텍스트에 있는 정보를 우선적으로 활용하세요.
                 2. 대화 이력을 참고하여 일관된 답변을 제공하세요.
                 3. 확실하지 않은 내용은 모른다고 답변하세요.
                 4. 질문에 대한 답변은 한국어로 제공하세요."""),

                 MessagesPlaceholder(variable_name="history"),
                 ("user", "{query}")
            ]).strip()