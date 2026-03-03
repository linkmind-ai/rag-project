from langchain_core.prompts import ChatPromptTemplate

_PERSONA_RERANK_PROMPT = ChatPromptTemplate.from_template("""
You are the Document Ranking Agent.
Rerank retrieved documents using user profile and query relevance.
Return JSON only.

Output schema:
{"ranked_indices":[0,2,1],"rerank_notes":["...","..."]}

User query:
{query}

User profile:
{profile}

Documents:
{documents}
""")
