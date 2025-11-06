from typing import Dict, List, Any
from dataclasses import dataclass, field
from langchain.schema import BaseMessage, HumanMessage, AIMessage

@dataclass
class ChatState:
    messages: List[BaseMessage] = field(default_factory=List)
    token_usage: Dict[str, int] = field(
        default_factory = lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    documents: List[Dict[str, Any]] = field(sefault_factory = list)

    def add_user_message(self, content:str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content:str) -> None:
        self.messages.append(AIMessage(content=content))

    def add_documents(self, content: str, doc_id: str) -> None:
        self.document.append({
            "doc_id": doc_id,
            "content": content
        })

    def add_usage(self, usage:Dict[str, int]) -> None:
        self.token_usage["input_tokens"] += usage.get("input_tokens", 0)
        self.token_usage["output_tokens"] += usage.get("output_tokens", 0)
        self.token_usage["total_tokens"] += usage.get("total_tokens", 0)

    def set_usage(self, usage:Dict[str, int]) -> None:
        self.token_usage = {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0)
        }

    def total_tokens(self) -> int:
        return self.token_usage.get("total_tokens", 0)
    
    def set_status(self, status:str):
        self.metadata["status"] = status
