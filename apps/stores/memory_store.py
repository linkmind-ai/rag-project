import asyncio
from datetime import datetime
from typing import Any

from common.config import settings
from models.state import ChatHistory, Message


class InMemoryStore:
    """Async in-memory store for sessions, profile, and feedback."""

    def __init__(self) -> None:
        self._store: dict[str, ChatHistory] = {}
        self._profiles: dict[str, dict[str, Any]] = {}
        self._feedback_events: dict[str, list[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def _default_profile(self) -> dict[str, Any]:
        return {
            "preferred_topics": [],
            "avoid_topics": [],
            "response_style": "balanced",
            "factuality_bias": 0.5,
            "last_feedback_rating": None,
            "explicit_notes": [],
            "updated_at": datetime.now().isoformat(),
        }

    async def get_history(self, session_id: str) -> ChatHistory:
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
        limit = limit or settings.MAX_HISTORY_LENGTH
        history = await self.get_history(session_id)
        async with self._lock:
            return history.get_recent_messages_history(limit)

    async def clear_history(self, session_id: str) -> bool:
        async with self._lock:
            removed = False
            if session_id in self._store:
                del self._store[session_id]
                removed = True
            if session_id in self._profiles:
                del self._profiles[session_id]
            if session_id in self._feedback_events:
                del self._feedback_events[session_id]
            return removed

    async def get_all_sessions(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())

    async def session_exists(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._store

    async def get_user_profile(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            if session_id not in self._profiles:
                self._profiles[session_id] = self._default_profile()
            return dict(self._profiles[session_id])

    async def update_user_profile(
        self, session_id: str, updates: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._lock:
            if session_id not in self._profiles:
                self._profiles[session_id] = self._default_profile()

            profile = self._profiles[session_id]
            profile.update(updates)
            profile["updated_at"] = datetime.now().isoformat()
            self._profiles[session_id] = profile
            return dict(profile)

    async def add_feedback_event(
        self, session_id: str, feedback: dict[str, Any]
    ) -> dict[str, Any]:
        event = {
            **feedback,
            "timestamp": datetime.now().isoformat(),
        }
        async with self._lock:
            if session_id not in self._feedback_events:
                self._feedback_events[session_id] = []
            self._feedback_events[session_id].append(event)
        return event

    async def get_feedback_events(self, session_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._feedback_events.get(session_id, []))

    async def update_profile_from_feedback(
        self, session_id: str, feedback: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._lock:
            if session_id not in self._profiles:
                self._profiles[session_id] = self._default_profile()

            profile = self._profiles[session_id]

            rating = feedback.get("rating")
            if isinstance(rating, int):
                profile["last_feedback_rating"] = rating
                if rating <= 2:
                    profile["factuality_bias"] = min(
                        1.0, float(profile.get("factuality_bias", 0.5)) + 0.1
                    )
                elif rating >= 4:
                    profile["factuality_bias"] = max(
                        0.0, float(profile.get("factuality_bias", 0.5)) - 0.05
                    )

            tags = feedback.get("tags") or []
            if isinstance(tags, list):
                preferred = set(profile.get("preferred_topics", []))
                for tag in tags:
                    if isinstance(tag, str) and tag.strip():
                        preferred.add(tag.strip())
                profile["preferred_topics"] = sorted(preferred)

            text = feedback.get("feedback_text")
            if isinstance(text, str) and text.strip():
                notes = list(profile.get("explicit_notes", []))
                notes.append(text.strip())
                profile["explicit_notes"] = notes[-20:]

            metadata = feedback.get("metadata") or {}
            if isinstance(metadata, dict):
                if isinstance(metadata.get("response_style"), str):
                    profile["response_style"] = metadata["response_style"]
                if isinstance(metadata.get("avoid_topics"), list):
                    profile["avoid_topics"] = metadata["avoid_topics"]

            profile["updated_at"] = datetime.now().isoformat()
            self._profiles[session_id] = profile
            return dict(profile)


memory_store = InMemoryStore()
