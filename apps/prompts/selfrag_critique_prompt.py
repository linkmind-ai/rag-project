from langchain_core.prompts import ChatPromptTemplate

_SELFRAG_CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You are the Self-RAG critique agent.
Evaluate the draft answer sentence-by-sentence.
Return JSON only.

Output schema:
{
  "retrieve_decisions":["No|Yes|Continue", "..."],
  "rel_scores":[0.0, 0.0],
  "support_labels":["fully|partial|none", "..."],
  "utility_score": 1.0,
  "insufficiency_hints":["missing_entity|missing_date|missing_definition|contradictory|irrelevant_docs"]
}

Query:
{query}

Draft answer:
{draft_answer}

Segments:
{segments}

Documents:
{documents}
""")
