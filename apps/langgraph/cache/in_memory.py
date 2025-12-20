import asyncio
from typing import Dict, Optional

from langgraph.state.chat_state import ChatState


class InMemoryStateCache:
    """
    멀티턴 RAG 시스템을 위한 인메모리 캐시
    - 세션별 ChatState 저장
    - FastAPI 라우터의 POST/GET 요청 고려
    - load_state, save_state, dump_state 지원
    """

    def __init__(self):
        # 세션별 저장소 (thread-safe를 위해 asyncio.Lock 사용)
        self._store: Dict[str, ChatState] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, session_id: str, state: ChatState) -> None:
        """
        세션의 ChatState 저장 (POST 스트리밍 서비스에서 사용)
        """
        async with self._lock:
            self._store[session_id] = state

    async def load_state(self, session_id: str) -> Optional[ChatState]:
        """
        세션의 ChatState 로딩 (GET /chat/stream/{session_id} 에서 사용)
        """
        async with self._lock:
            return self._store.get(session_id)

    async def dump_state(self, session_id: str) -> Optional[dict]:
        """
        state 내용을 dict으로 반환
        (디버깅, 관리 목적)
        """
        async with self._lock:
            state = self._store.get(session_id)
            return state.dict() if state else None

    async def reset_state(self, session_id: str) -> None:
        """
        특정 세션 상태 삭제
        (대화 초기화 기능)
        """
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]

    async def list_sessions(self) -> Dict[str, int]:
        """
        현재 저장된 세션 목록 조회 (디버깅용)
        """
        async with self._lock:
            return {sid: len(st.messages) for sid, st in self._store.items()}