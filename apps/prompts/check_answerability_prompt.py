from langchain_core.prompts import ChatPromptTemplate

_CHECK_ANSWERABILITY_PROMPT = ChatPromptTemplate.from_template("""
        당신은 주어진 질문과 그래프 형태의 정보로 해당 질문에 답변할 수 있는지 판단하는 어시스턴트 입니다.
        아래 주어진 지식 그래프를 가지고 질문에 답할 수 있는지 판단하세요.
        
        질문: {query}
                
        지식 그래프: {triples}
                 
        답변 가능 여부를 판단하고 그 이유를 설명하세요. 응답은 아래와 같이 JSON 형태로 반환하세요.
                 
        응답 형식 예시(JSON 형식으로 반환):
        {{"answerable": True/False, "reasoning": "판단 이유"}}
        """)
