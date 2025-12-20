from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from langchain.schema import BaseMessage, HumanMessage, AIMessage

@dataclass
class ChatState:
    messages: List[BaseMessage] = field(defult_factory = list)
    token_usage: Dict[str, int] = field(
        default_factory = lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    documents: List[Dict[str, Any]] = field(default_factory = list)
    answer: str = ""
    references = List[Dict[str, Any]] = field(default_factory = list)
    metadata: Dict[str, Any] = field(default_factory = dict)

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content = content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content = content))

    def user_latest_message(self) -> str:
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""
    
    def add_usage(self, usage: Dict[str, int]) -> None:
        self.token_usage = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }

    def total_tokens(self) -> int:
        return self.token_usage.get("total_tokens", 0)
    
    def set_status(self, status: str):
        self.metadata["status"] = status

    def update_response(self, response: Dict[str, Any]) -> None:
        self.answer = response.get("answer", "")
        self.references = response.get("references", [])

    def add_documents(self, document: Dict[str, Any]) -> None:
        self.documents.append(document)