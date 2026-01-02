import asyncio
from typing import List, Dict, Any, Optional
import aiohttp

from models import Document
from config import settings


class NotionConnector:
    """비동기 노션API 연동 클래스"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "Authorization": f"Bearer {settings.NOTION_TOKEN}",
            "Notion-Version": settings.NOTION_VERSION,
            "Content-Type": "application/json"
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """세션 연결"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """세션 종료"""
        if self._session and not self._session.closed:
            await self._session.close() 

    async def get_page_content(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        페이지 ID를 기반으로 페이지 내용을 가져옵니다.

        Args:
            page_id (str): Notion 페이지 ID (대시 제외)

        Returns:
            dict: 페이지 속성 및 내용
        """
        session = await self._get_session()

        # 페이지 속성 가져오기
        page_url = f"https://api.notion.com/v1/pages/{page_id}"

        try:
            async with session.get(page_url, headers=self._headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"페이지 가져오기 오류: {response.status}")
                    print(error_text)
                    return None
                
                page_data = await response.json()

        except Exception as e:
            print(f"페이지 요청 오류: {str(e)}")
            return None


        # 페이지 내용(블록) 가져오기
        blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"

        try:
            async with session.get(blocks_url, headers=self._headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"블록 가져오기 오류: {response.status}")
                    print(error_text)
                    return page_data
                
                blocks_data = await response.json()
                page_data["content"] = blocks_data["results"]

        except Exception as e:
            print(f"블록 요청 오류: {str(e)}")
            
        return page_data

    async def fetch_all_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """
        블록의 모든 하위 블록을 재귀적으로 가져옵니다.

        Args:
            block_id (str): 블록 ID

        Returns:
            list: 모든 하위 블록 목록
        """
        session = await self._get_session()
        blocks_url = f"https://api.notion.com/v1/blocks/{block_id}/children"

        try:
            async with session.get(blocks_url, headers=self._headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"블록 가져오기 오류: {response.status}")
                    print(error_text)
                    return []]
                
                blocks_data = await response.json()
                blocks = blocks_data.get("results", [])

        except Exception as e:
            print(f"블록 요청 오류: {str(e)}")
            return []

        # 하위 블록이 있는 블록 타입들
        has_children_types = [
            "paragraph", "bulleted_list_item", "numbered_list_item",
            "toggle", "child_page", "child_database", "column_list",
            "column", "table", "synced_block"
        ]

        all_blocks = []

        for block in blocks:
            all_blocks.append(block)

            # 하위 블록이 있는 경우 재귀적으로 가져오기
            if block.get("has_children") and block.get("type") in has_children_types:
                children = await self.fetch_all_blocks(block["id"])
                # 하위 블록이 있는 경우에만 children 키 추가
                if children:
                    block["children"] = children

        return all_blocks

    def extract_text_content(self, blocks: List[Dict[str, Any]]) -> str:
        """
        블록에서 텍스트 내용만 추출합니다. 이때, child_page의 제목을 구분자로 먼저 추가합니다.

        Args:
            blocks (list): 블록 목록

        Returns:
            str: 추출된 텍스트 내용
        """
        text_content = ""

        for block in blocks:
            block_type = block.get("type")

            # child_page의 경우, 제목 먼저 삽입
            if block_type == "child_page":
                child_title = block.get("child_page", {}).get("title", "")
                if child_title:
                    text_content += f"\n\n### {child_title} ###\n\n"

            if block_type == "paragraph":
                rich_text = block.get("paragraph", {}).get("rich_text", [])
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "heading_1":
                rich_text = block.get("heading_1", {}).get("rich_text", [])
                for text in rich_text:
                    text_content += "# " + text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "heading_2":
                rich_text = block.get("heading_2", {}).get("rich_text", [])
                for text in rich_text:
                    text_content += "## " + text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "heading_3":
                rich_text = block.get("heading_3", {}).get("rich_text", [])
                for text in rich_text:
                    text_content += "### " + text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "bulleted_list_item":
                rich_text = block.get("bulleted_list_item", {}).get("rich_text", [])
                text_content += "• "
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n"

            elif block_type == "numbered_list_item":
                rich_text = block.get("numbered_list_item", {}).get("rich_text", [])
                text_content += "1. "
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n"

            elif block_type == "to_do":
                rich_text = block.get("to_do", {}).get("rich_text", [])
                checked = block.get("to_do", {}).get("checked", False)
                text_content += "[{}] ".format("x" if checked else " ")
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n"

            elif block_type == "code":
                rich_text = block.get("code", {}).get("rich_text", [])
                language = block.get("code", {}).get("language", "")
                text_content += f"```{language}\n"
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n```\n\n"

            elif block_type == "quote":
                rich_text = block.get("quote", {}).get("rich_text", [])
                text_content += "> "
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "callout":
                rich_text = block.get("callout", {}).get("rich_text", [])
                icon = block.get("callout", {}).get("icon", {})
                emoji = icon.get("emoji", "+++") if icon.get("type") == "emoji" else "+++"
                text_content += "{emoji} "
                for text in rich_text:
                    text_content += text.get("plain_text", "")
                text_content += "\n\n"

            elif block_type == "divider":
                text_content += "\n\n"

            # 자식 블록이 있으면 재귀적으로 처리
            if "children" in block:
                text_content += self.extract_text_content(block["children"])

        return text_content

    def _extract_page_title(self, page_data: Dict[str, Any]) -> str:
        """페이지 제목 추출"""
        properties = page_data.get("properties", {})
        
        title_keys = ["title", "Title", "Name", "name"]
        
        for key in title_keys:
            if key in properties:
                title_prop = properties[key]
                if title_prop.get("type") == "title":
                    title_array = title_prop.get("title", [])
                    if title_array:
                        return "".join([t.get("plain_text", "") for t in title_array])
        
        return "Untitled"
    
    async def fetch_page_as_document(
        self, 
        page_id: str, 
        recursive: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """
        Notion 페이지를 Document 객체로 변환합니다.

        Args:
            page_id: Notion 페이지 ID
            recursive: 하위 블록을 재귀적으로 가져올지 여부
            additional_metadata: 추가 메타데이터

        Returns:
            Document 객체
        """
        page_data = await self.get_page_content(page_id)
        
        if not page_data:
            return None
        
        page_title = self._extract_page_title(page_data)
        
        blocks = page_data.get("content", [])
        
        if recursive:
            all_blocks = []
            for block in blocks:
                all_blocks.append(block)
                if block.get("has_children"):
                    children = await self.fetch_all_blocks(block["id"])
                    if children:
                        block["children"] = children
            blocks = all_blocks
        
        text_content = self.extract_text_content(blocks)
        
        metadata = {
            "source": "notion",
            "page_id": page_id,
            "page_title": page_title,
            "page_url": f"https://www.notion.so/{page_id.replace('-', '')}",
            "created_time": page_data.get("created_time", ""),
            "last_edited_time": page_data.get("last_edited_time", "")
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        document = Document(
            content=text_content,
            metadata=metadata
        )
        
        return document


notion_connector = NotionConnector()