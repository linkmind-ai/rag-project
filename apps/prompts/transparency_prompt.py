from langchain_core.prompts import ChatPromptTemplate

_TRANSPARENCY_PROMPT = ChatPromptTemplate.from_template("""
Write a short transparency explanation for the user.
Mention:
1) what profile preference was prioritized,
2) whether additional retrieval was triggered and why,
3) how final evidence was selected.

Query:
{query}

Route:
{route}

Loop count:
{loop_count}

Insufficiency reasons:
{reasons}

Rerank notes:
{rerank_notes}
""")
