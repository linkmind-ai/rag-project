from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

from common.config import settings
from models.state import Document

logger = logging.getLogger(__name__)


class NotionConnector:
    """비동기 Notion API 연동 클라이언트.

    - 세션/헤더 속성명을 _session/_headers로 단일화합니다.
    - 모든 I/O 요청에 timeout을 명시합니다.
    - print 대신 구조화 로그(extra)를 사용합니다.
    - 세션은 지연 생성(lazy)하며, close()로 종료 가능합니다.
    """

    BASE_URL = "https://api.notion.com/v1"

    def __init__(
        self,
        notion_token: str,
        notion_version: str,
        timeout_seconds: float = 15.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._session: Optional[aiohttp.ClientSession] = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._headers: Dict[str, str] = {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": notion_version,
            "Content-Type": "application/json",
        }

    async def __aenter__(self) -> "NotionConnector":
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """aiohttp 세션을 지연 생성하여 반환합니다."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """세션 종료"""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_json(self, url: str) -> Optional[Dict[str, Any]]:
        """GET 요청 후 JSON 응답을 반환합니다."""
        session = await self._get_session()

        try:
            async with session.get(url, headers=self._headers) as response:
                if response.status < 200 or response.status >= 300:
                    error_text = await response.text()
                    logger.warning(
                        "notion_api_non_2xx",
                        extra={
                            "url": url,
                            "status": response.status,
                            "error_text": error_text[:1000],
                        },
                    )
                    return None
                return await response.json()
        except aiohttp.ClientError as exc:
            logger.exception(
                "notion_api_client_error",
                extra={"url": url, "error": str(exc)},
            )
            return None

    async def get_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """페이지 ID 기반으로 페이지 속성과 루트 블록(children)을 가져옵니다."""
        page_url = f"{self.BASE_URL}/pages/{page_id}"
        page_data = await self._get_json(page_url)
        if not page_data:
            return None

        blocks_url = f"{self.BASE_URL}/blocks/{page_id}/children"
        blocks_data = await self._get_json(blocks_url)
        if blocks_data and isinstance(blocks_data.get("results"), list):
            page_data["content"] = blocks_data["results"]
        else:
            page_data["content"] = []

        return page_data

    async def fetch_all_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """block_id의 모든 하위 블록을 재귀적으로 가져옵니다."""
        blocks_url = f"{self.BASE_URL}/blocks/{block_id}/children"
        blocks_data = await self._get_json(blocks_url)
        if not blocks_data:
            return []

        blocks = blocks_data.get("results", [])
        if not isinstance(blocks, list):
            return []

        all_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue

            all_blocks.append(block)
            if block.get("has_children") is True and isinstance(block.get("id"), str):
                children = await self.fetch_all_blocks(block["id"])
                if children:
                    block["children"] = children

        return all_blocks

    def extract_text_content(self, blocks: List[Dict[str, Any]]) -> str:
        """Notion 블록 목록에서 텍스트만 추출합니다."""
        parts: List[str] = []
        self._append_blocks_text(parts, blocks, ordered_index_start=1)
        return "".join(parts)

    def _append_blocks_text(
        self,
        parts: List[str],
        blocks: Iterable[Dict[str, Any]],
        ordered_index_start: int,
    ) -> None:
        ordered_index = ordered_index_start

        for block in blocks:
            block_type = block.get("type")

            # child_page의 경우, 제목 먼저 삽입
            if block_type == "child_page":
                child_title = block.get("child_page", {}).get("title")
                if isinstance(child_title, str) and child_title:
                    parts.append(f"\n\n### {child_title} ###\n\n")

            if block_type == "paragraph":
                self._append_rich_text(
                    parts, block.get("paragraph", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_1":
                parts.append("# ")
                self._append_rich_text(
                    parts, block.get("heading_1", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_2":
                parts.append("## ")
                self._append_rich_text(
                    parts, block.get("heading_2", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_3":
                parts.append("### ")
                self._append_rich_text(
                    parts, block.get("heading_3", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "bulleted_list_item":
                parts.append("• ")
                self._append_rich_text(
                    parts, block.get("bulleted_list_item", {}).get("rich_text", [])
                )
                parts.append("\n")

            elif block_type == "numbered_list_item":
                parts.append(f"{ordered_index}. ")
                self._append_rich_text(
                    parts, block.get("numbered_list_item", {}).get("rich_text", [])
                )
                parts.append("\n")
                ordered_index += 1

            elif block_type == "to_do":
                todo = block.get("to_do", {})
                checked = bool(todo.get("checked", False))
                parts.append(f"[{'x' if checked else ' '}] ")
                self._append_rich_text(parts, todo.get("rich_text", []))
                parts.append("\n")

            elif block_type == "code":
                code = block.get("code", {})
                language = code.get("language")
                parts.append(f"```{language if isinstance(language, str) else ''}\n")
                self._append_rich_text(parts, code.get("rich_text", []))
                parts.append("\n```\n\n")

            elif block_type == "quote":
                parts.append("> ")
                self._append_rich_text(
                    parts, block.get("quote", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "callout":
                callout = block.get("callout", {})
                icon = callout.get("icon", {})
                emoji = "+++"
                if (
                    isinstance(icon, dict)
                    and icon.get("type") == "emoji"
                    and isinstance(icon.get("emoji"), str)
                ):
                    emoji = icon["emoji"]
                parts.append(f"{emoji} ")
                self._append_rich_text(parts, callout.get("rich_text", []))
                parts.append("\n\n")

            elif block_type == "divider":
                parts.append("\n---\n\n")

            children = block.get("children")
            if isinstance(children, list) and children:
                self._append_blocks_text(parts, children, ordered_index_start=1)

    def _append_rich_text(self, parts: List[str], rich_text: Any) -> None:
        if not isinstance(rich_text, list):
            return
        for item in rich_text:
            if isinstance(item, dict):
                text = item.get("plain_text")
                if isinstance(text, str) and text:
                    parts.append(text)

    def _extract_page_title(self, page_data: Dict[str, Any]) -> str:
        """페이지 제목 추출"""
        properties = page_data.get("properties", {})
        if not isinstance(properties, dict):
            return "Untitled"

        for key in ("title", "Title", "Name", "name"):
            title_prop = properties.get(key)
            if not isinstance(title_prop, dict):
                continue
            if title_prop.get("type") != "title":
                continue
            title_array = title_prop.get("title", [])
            if not isinstance(title_array, list) or not title_array:
                continue
            return "".join(
                t.get("plain_text", "") for t in title_array if isinstance(t, dict)
            )

        return "Untitled"

    async def fetch_page_as_document(
        self,
        page_id: str,
        recursive: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """Notion 페이지를 Document 객체로 변환합니다."""
        page_data = await self.get_page_content(page_id)
        if not page_data:
            return None

        page_title = self._extract_page_title(page_data)

        blocks = page_data.get("content", [])
        if not isinstance(blocks, list):
            blocks = []

        if recursive:
            all_blocks: List[Dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                all_blocks.append(block)
                if block.get("has_children") is True and isinstance(
                    block.get("id"), str
                ):
                    children = await self.fetch_all_blocks(block["id"])
                    if children:
                        block["children"] = children
            blocks = all_blocks

        text_content = self.extract_text_content(blocks)

        metadata: Dict[str, Any] = {
            "source": "notion",
            "page_id": page_id,
            "page_title": page_title,
            "page_url": f"https://www.notion.so/{page_id.replace('-', '')}",
            "created_time": page_data.get("created_time", ""),
            "last_edited_time": page_data.get("last_edited_time", ""),
        }
        if additional_metadata:
            metadata.update(additional_metadata)

        return Document(
            content=text_content,
            metadata=metadata,
            doc_id=f"notion_{page_id.replace('-', '')}",
        )


def build_notion_connector() -> NotionConnector:
    """설정값 기반으로 NotionConnector를 생성합니다."""
    return NotionConnector(
        notion_token=settings.NOTION_TOKEN,
        notion_version=settings.NOTION_VERSION,
        timeout_seconds=15.0,
    )


# 싱글톤 인스턴스
notion_connector = build_notion_connector()
