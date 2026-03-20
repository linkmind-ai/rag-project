from langchain_core.prompts import ChatPromptTemplate

_CONTEXT_ATTACHMENT_PROMPT = ChatPromptTemplate.from_template("""
You decide whether the current search query needs recent user context attached.
Return JSON only.

Output schema:
{{
  "attach_context": true,
  "reason": "short explanation"
}}

Rules:
- Set attach_context to true only when the current question depends on omitted referents,
  prior user intent, or earlier user constraints to be searchable.
- Set attach_context to false when the current question is already standalone and specific enough.
- Use recent user context only. Do not rely on assistant messages as evidence.
- Be conservative: if attaching context would mostly add noise, return false.

Current question:
{query}

Recent user context:
{recent_user_context}

Session summary:
{session_summary}

Preferred topics:
{preferred_topics}
""")
