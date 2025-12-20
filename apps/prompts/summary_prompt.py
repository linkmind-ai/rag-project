import textwrap
from langchain_core.prompts import PromptTemplate

_SUMMARY_PROMPT = textwrap.dedent("""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

당신은 주어진 정보에서 핵심 정보를 기반으로 대화의 역사를 요약 관리하는 담당자입니다.
                                  
주어진 대화 내역에서 핵심 정보를 구조화하여 응답하세요.

<|eot_id|>

<|start_header_id|>user<|end_header_id|>
                               
<task>
다음의 내화 내역을 요약해주세요.
</task>
                                  
<requirements>
- 항상 plain text 만을 사용하여 markdown 없이 요약해주세요.
- 주요한 객체를 중심으로 리스트로 요약해주세요.
- 포괄적인 범위에서 주제 별로 정리해주세요.
- 최대 200자를 넘지 않도록 해주세요.
</requirements>
                               
{chat_history}
                               
<|eot_id|>                            

<|start_header_id|>assistant<|end_header_id|>

답변:                         
""").strip()

SUMMARY_PROMPT = PromptTemplate.from_template(_SUMMARY_PROMPT)
                                                              