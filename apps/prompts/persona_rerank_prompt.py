from langchain_core.prompts import ChatPromptTemplate

_PERSONA_RERANK_PROMPT = ChatPromptTemplate.from_template("""
You are an information organization expert aiding the Document Ranking Agent in prioritizing information.

Question:
{query}

Passages:
{documents}

Global Memory:
{global_memory}

Task Description:
Review the retrieved passages and rank them from most to least relevant by considering the user's stable preferences, recent session context, and current intent. Prioritize passages that best support a personalized and contextually appropriate answer.

Return JSON only.
Output schema:
{"ranked_indices":[0,2,1],"rerank_notes":["...","..."]}
""")
