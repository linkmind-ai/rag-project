import asyncio

from common.config import settings
from models.state import ChatHistory, Message


class InMemoryStore:
    """비동기 인메모리 저장소"""

    def __init__(self):
        self._store: dict[str, ChatHistory] = {}
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> ChatHistory:
        """세션 대화 이력 불러오기"""
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)
            return self._store[session_id]

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)

            history = self._store[session_id]
            history.add_message(role, content)

            if len(history.messages) > settings.MAX_HISTORY_LENGTH:
                history.messages = history.messages[-settings.MAX_HISTORY_LENGTH :]

    async def get_recent_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[Message]:
        """최근 메시지 불러오기"""
        limit = limit or settings.MAX_HISTORY_LENGTH
        history = await self.get_history(session_id)
        async with self._lock:
            return history.get_recent_messages_history(limit)

    async def clear_history(self, session_id: str) -> bool:
        """이력 삭제"""
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    async def get_all_sessions(self) -> list[str]:
        """모든 세션 ID 목록 불러오기"""
        async with self._lock:
            return list(self._store.keys())

    async def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        async with self._lock:
            return session_id in self._store


memory_store = InMemoryStore()
