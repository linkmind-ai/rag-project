from pydantic import BaseModel, Field, validator


class ChunkItem(BaseModel):
    title: str = Field(..., description="Chunk title extracted from ### HEADER ###")
    content: str = Field(..., description="Chunk content text")
    keywords: str = Field(..., description="Extracted keywords joined by | delimiter")

    @validator("title", "content", "keywords")
    def validate_not_empty(cls, value, field):
        """빈 문자열이나 공백만 있는 경우 방지"""
        if not isinstance(value, str):
            raise TypeError(f"{field.name} must be a string")

        if not value.strip():
            raise ValueError(f"{field.name} cannot be empty")

        return value