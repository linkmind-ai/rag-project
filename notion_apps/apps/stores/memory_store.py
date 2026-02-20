import asyncio
from typing import Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from common.config import settings
from models.state import ChatHistory


class InMemoryStore:
    """비동기 인메모리 저장소"""

    def __init__(self):
        self._store: Dict[str, ChatHistory] = {}
        self._lock = asyncio.Lock()

    async def get_history(self, session_id: str) -> ChatHistory:
        """세션 대화 이력 불러오기"""
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)
            return self._store[session_id]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:
        """
        role 문자열 기반 메시지 추가
         ChatHistory.add_message가 내부적으로 role → BaseMessage 변환 처리
        """
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)

            history = self._store[session_id]
            history.add_message(role, content)

            #  history.messages는 List[BaseMessage]이므로 슬라이싱 정상 동작
            if len(history.messages) > settings.MAX_HISTORY_LENGTH:
                history.messages = history.messages[-settings.MAX_HISTORY_LENGTH:]

    async def add_base_message(
        self,
        session_id: str,
        message: BaseMessage
    ) -> None:
        """
        BaseMessage 직접 추가
         LangGraph 노드 출력 결과를 바로 저장할 때 사용
        """
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)

            history = self._store[session_id]
            history.messages.append(message)

            if len(history.messages) > settings.MAX_HISTORY_LENGTH:
                history.messages = history.messages[-settings.MAX_HISTORY_LENGTH:]

    async def save_interaction(
        self,
        session_id: str,
        user_query: str,
        ai_answer: str
    ) -> None:
        """
        사용자 질문 + AI 응답 한번에 저장
         RAG 파이프라인 완료 후 호출하여 대화 이력 누적
           HumanMessage / AIMessage로 각각 변환하여 저장
        """
        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)

            history = self._store[session_id]
            history.messages.append(HumanMessage(content=user_query))
            history.messages.append(AIMessage(content=ai_answer))

            if len(history.messages) > settings.MAX_HISTORY_LENGTH:
                history.messages = history.messages[-settings.MAX_HISTORY_LENGTH:]

    async def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Sequence[BaseMessage]:
        """
        최근 메시지 불러오기
         반환 타입: list[Message] → Sequence[BaseMessage]
           GraphState.chat_history 타입과 일치하므로 바로 할당 가능

         데드락 버그 수정: 기존 코드는 get_history() 내부에서 lock을 획득한 상태로
           반환된 뒤, 다시 get_recent_messages()에서 lock을 획득 시도하여
           asyncio.Lock()의 non-reentrant 특성상 데드락이 발생할 수 있었음
           → lock 없이 직접 _store 접근하도록 수정
        """
        limit = limit or settings.MAX_HISTORY_LENGTH

        async with self._lock:
            if session_id not in self._store:
                self._store[session_id] = ChatHistory(session_id=session_id)
            history = self._store[session_id]
            return history.get_recent_messages_history(limit)

    async def clear_history(self, session_id: str) -> bool:
        """이력 삭제"""
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    async def get_all_sessions(self) -> List[str]:
        """모든 세션 ID 목록 불러오기"""
        async with self._lock:
            return list(self._store.keys())

    async def session_exists(self, session_id: str) -> bool:
        """세션 존재 여부 확인"""
        async with self._lock:
            return session_id in self._store


memory_store = InMemoryStore()
