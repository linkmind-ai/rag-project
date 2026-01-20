from langchain.prompts import ChatPromptTemplate


_GET_EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([
                ("system", """당신은 주어진 질문에 답변하기 위해 제공된 문서 중 실제 응답에 사용된 문서를 선택하는 근거 문서 분석 전문가입니다.
                 제공된 질문과 생성된 응답, 검색된 문서를 바탕으로 생성된 응답에 실제 사용된 근거 문서를 선택해주세요.
                 
                 질문: {query}
                
                 생성된 답변: {answer}

                 검색된 문서들:
                 {documents}
                 
                 위 답변을 생성할 때 근거로 사용된 문서의 인덱스를 JSON 배열 형식으로만 반환하세요.
                 
                 응답 형식 예시(string이 아닌 JSON 형식으로 반환):
                 {{"evidence_indices": [0, 2, 3]}}""")
            ]).strip()