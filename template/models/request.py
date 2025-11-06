from typing import Literal
from pydantic import BaseModel, Field
from models.base import ChatMessage

class ChatRequest(BaseModel):
    user_id: Optional[str] = Field("anonymous", description = "요청 사용자 ID")
    session_id: str = Field(..., description = "세션 ID")
    message: str = Field(..., description = "사용자 입력 문장")
    history: Optional[List[ChatMessage]] = Field(default_factory = list, description = "이전 대화 기록")