from typing import TypedDict, List, Optional

class RAGState(TypedDict):
    question: str
    expanded_query: Optional[str]
    contexts: Optional[List[str]]
    answer: Optional[str]