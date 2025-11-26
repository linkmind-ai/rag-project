INITIAL_RESPONSE_PROMPT = """
                        다음 question에 대한 요약된 답변을 작성해줘.
                        question:
                        {question}
                        """
                        
FINAL_RESPONSE_PROMPT = """
                        다음 question에 대해 context에 기반해서 답변해줘. 
                        단, 'context에 기반한 ~'과 같은 표현은 사용하지 마.
                        question:
                        {question}
                        contexts:
                        {contexts}
                        """