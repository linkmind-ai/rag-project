from langchain_core.prompts import ChatPromptTemplate

_SELFRAG_DRAFT_PROMPT = ChatPromptTemplate.from_template("""
You are drafting an answer from personalized evidence.
Use E0 and M0 faithfully. If evidence is weak, state uncertainty.

User query:
{query}

E0 docs:
{documents}

E0 summaries:
{doc_summaries}

M0 global message pool:
{global_pool}

Write a concise answer:
""")
