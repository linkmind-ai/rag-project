from langchain_core.prompts import ChatPromptTemplate

_SELFRAG_REWRITE_PROMPT = ChatPromptTemplate.from_template("""
You are rewriting a retrieval query because current evidence is insufficient.
Return JSON only.

Output schema:
{"rewritten_query":"...","focus":["..."]}

Original query:
{query}

Insufficiency reasons:
{reasons}

Current plan:
{retrieval_plan}
""")
