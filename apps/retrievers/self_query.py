import asyncio
from typing import Any

from common.config import settings
from common.logger import logger
from langchain.chains.query_constructor.base import (
    AttributeInfo,
    load_query_constructor_runnable,
)
from langchain_community.query_constructors.elasticsearch import ElasticsearchTranslator
from langchain_community.llms import Ollama
from langchain_core.runnables import Runnable


class MetadataQueryRetriever:
    """LLM이 자연어 쿼리를 (검색어, 메타데이터 필터)로 파싱하는 컴포넌트."""

    def __init__(self) -> None:
        self.metadata_field_info: list[AttributeInfo] = [
            AttributeInfo(
                name="page_title",
                description="문서 내용이 포함된 Notion 페이지의 제목명",
                type="string",
            )
        ]
        self.document_contents: str = (
            "Notion 워크스페이스에 저장된 개인 기록 및 문서 모음"
        )

        self._llm: Ollama | None = None
        self._query_constructor: Runnable | None = None
        self._translator: ElasticsearchTranslator = ElasticsearchTranslator()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._initialized: bool = False

    async def initialize(self) -> None:
        """LLM과 query_constructor를 지연 생성 (Double-Checked Locking)."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            self._llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.REWRITER_MODEL,
                headers={
                    "CF-Access-Client-Id": settings.CF_ACCESS_CLIENT_ID,
                    "CF-Access-Client-Secret": settings.CF_ACCESS_CLIENT_SECRET,
                },
                temperature=0.0,
            )

            self._query_constructor = load_query_constructor_runnable(
                self._llm,
                self.document_contents,
                self.metadata_field_info,
            )

            self._initialized = True

    async def parse(self, query: str) -> tuple[str, dict[str, Any] | None]:
        """자연어 쿼리를 (재작성된 검색어, ES 메타데이터 필터 dict)로 파싱.

        Args:
            query: 사용자 자연어 질의

        Returns:
            (rewritten_query, filter_dict)
            - rewritten_query: 메타데이터 조건이 제거된 순수 검색어
            - filter_dict: ES bool/filter 절에 그대로 넣을 수 있는 dict, 필터가 없으면 None
        """
        await self.initialize()

        if self._query_constructor is None:
            raise RuntimeError("MetadataQueryRetriever가 초기화되지 않았습니다.")

        try:
            structured_query = await self._query_constructor.ainvoke({"query": query})
            new_query, search_kwargs = self._translator.visit_structured_query(
                structured_query
            )
            return new_query, search_kwargs.get("filter")
        except Exception as exc:
            # 메타데이터 파싱 실패는 보조 기능 실패이므로 원 쿼리로 폴백
            logger.warning(
                "[MetadataQueryRetriever] 파싱 실패, 원 쿼리로 폴백: {}", exc
            )
            return query, None


metadata_query_retriever = MetadataQueryRetriever()
