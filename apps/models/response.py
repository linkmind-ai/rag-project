from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from models.base import ChatMessage, TokenUsage

class ChatStateResponse(BaseModel):
    session_id: str
    message: List[ChatMessage]
    token_usage: TokenUsage = Field(default_factory = lambda:TokenUsage())

    model_config = ConfigDict(
        json_schema_extra = {
            "examples": [
                {
                 "session_id": "",
                 "messages": [],
                 "token_usage": {
                     "input_tokens": 0,
                     "output_tokens": 0,
                     "total_tokens": 0
                 }   
                }
            ]
        }
    )