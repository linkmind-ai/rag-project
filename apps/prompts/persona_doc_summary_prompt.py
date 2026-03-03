from langchain_core.prompts import ChatPromptTemplate

_PERSONA_DOC_SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are the PersonaRAG summarization agent.
Summarize each document with citation metadata.
Return JSON only.

Output schema:
{"summaries":[{"doc_index":0,"summary":"...","citation":"...","confidence":0.0}]}

Query:
{query}

Documents:
{documents}
""")
