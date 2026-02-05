from langchain_core.prompts import ChatPromptTemplate

_EXTRACT_TRIPLES_PROMPT = ChatPromptTemplate.from_template("""
            당신은 주어진 문서 내용 중 주어, 관계, 목적어를 분석하여 추출하는 전문가입니다.
            다음 문서들에서 질문과 관련된 지식 그래프 triples(주어-관계-목적어)를 추출하세요.
                 
            질문: {query}
                
            문서들:
            {passages}
            {existing_info}

            triples 추출 규칙:
            1. 질문에 답하는데 필요한 핵심 관계만 추출하세요.
            2. 각 triples는 (주어, 관계, 목적어)형태로 작성하세요.
            3. 구체적이고 명확하게 관계를 표현하세요.
            4. 최대 10개까지 추출하세요.
                 
            JSON 배열 형식으로만 반환하세요. 응답 형식 예시는 아래와 같습니다:
             {{"triples": [
               {{"subject": "주어", "relation": "관계", "object": "목적어"}},
                  ...
             ]}}
            """)
