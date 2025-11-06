from langchain.schema import BaseMessage, HumanMessage, AIMessage

from state.chat_state import ChatState
from models.base import ChatMessage, TokenUsage
from models.response import ChatStateResponse

class InMemoryCache:
    def __init__(self):
        self._store = {}

    async def load_state(self, session_id: str) -> ChatState:
        return self._store.get(session_id, ChatState())
    
    async def save_state(self, session_id: str, state: ChatState):
        self._store[session_id] = state

    async def dump_state(self, session_id: str) -> ChatStateResponse:
        state = self._store.get(session_id)
        if not state:
            return ChatStateResponse(
                session_id = session_id,
                messages = []
                token_usage = TokenUsage()
            )
        
        def to_msg(m: BaseMessage):
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            return ChatMessage(role = role, content = m.content)
        
        usage = TokenUsage(**state.token_usage)
        return ChatStateResponse(
            session_id = session_id,
            messages = [to_msg(m) for m in state.messages],
            token_usage = usage
        )