from langchain_core.prompts import ChatPromptTemplate

_SELFRAG_CRITIQUE_PROMPT = ChatPromptTemplate.from_template("""
You are evaluating a draft answer with the Self-RAG reflection criteria.
For each segment, judge using the following labels from Self-RAG:
- retrieval_decisions: [Retrieval] | [No Retrieval] | [Continue to Use Evidence]
- relevance_labels: [Relevant] | [Irrelevant]
- support_labels: [Fully supported] | [Partially supported] | [No support]
- utility_score: integer 1 to 5, corresponding to [Utility:1] ... [Utility:5]

Return JSON only.
Output schema:
{
  "retrieve_decisions":["[No Retrieval]","[Retrieval]"],
  "relevance_labels":["[Relevant]","[Irrelevant]"],
  "rel_scores":[1.0,0.0],
  "support_labels":["[Fully supported]","[Partially supported]"],
  "utility_score": 4,
  "insufficiency_hints":["missing_entity","missing_date","missing_definition","contradictory","irrelevant_docs"]
}

Question:
{query}

Draft answer:
{draft_answer}

Segments:
{segments}

Documents:
{documents}
""")
