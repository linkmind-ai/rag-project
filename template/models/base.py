from typing import Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class TokenUsage(BaseModel):
    input_tokens: int = Field(0, description = "입력 프롬프트 토큰 수")
    output_tokens: int = Field(0, description = "출력 프롬프트 토큰 수")
    total_tokens: int = Field(0, description = "총 토큰 수")