from langchain_core.prompts import ChatPromptTemplate


_REWRITE_QUERY_PROMPT = ChatPromptTemplate.from_template("""
               당신은 주어진 질문과 현재 정보를 가지고 질문을 재작성하는 작문가입니다.
               제공된 원본 질문과 현재 정보, 답변 불가한 이유를 바탕으로 원본 질문을 한 문장으로 재작성해주세요.
                 
               원본 질문: {query}
                
               현재 정보: {triples}

               답변 불가 이유:
               {reasoning}
                 
               재작성된 질문:
               """)