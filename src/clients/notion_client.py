# src/clients/notion_client.py

import requests
import logging

logger = logging.getLogger(__name__)

class NotionClient:
    """Notion API와 통신하여 데이터를 추출하는 클래스"""

    def __init__(self, token: str, version: str):
        if not token:
            raise ValueError("Notion 토큰이 필요합니다.")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Notion-Version": version,
            "Content-Type": "application/json"
        }
        self.has_children_types = [
            "paragraph", "bulleted_list_item", "numbered_list_item",
            "toggle", "child_page", "child_database", "column_list",
            "column", "table", "synced_block"
        ]

    def fetch_all_blocks(self, block_id: str) -> list:
        """블록의 모든 하위 블록을 재귀적으로 가져옵니다."""
        all_blocks = []
        blocks_url = f"https://api.notion.com/v1/blocks/{block_id}/children"

        try:
            response = requests.get(blocks_url, headers=self.headers)
            response.raise_for_status()

            blocks_data = response.json()["results"]

            for block in blocks_data:
                all_blocks.append(block)
                if block.get("has_children") and block.get("type") in self.has_children_types:
                    children = self.fetch_all_blocks(block["id"])
                    if children:
                        block["children"] = children
            return all_blocks

        except requests.exceptions.RequestException as e:
            logger.error(f"블록 가져오기 오류 (ID: {block_id}): {e}", exc_info=True)
            return []

    def extract_text_content(self, blocks: list) -> str:
        """블록 목록에서 텍스트 내용만 재귀적으로 추출합니다."""
        text_content = ""
        for block in blocks:
            block_type = block.get("type")
            text_to_add = ""

            if block_type == "child_page":
                child_title = block.get("child_page", {}).get("title", "")
                if child_title:
                    text_content += f"\n\n### {child_title} ###\n\n"

            elif block_type == "heading_1":
                text_to_add = "# " + "".join(
                    [t.get("plain_text", "") for t in block.get("heading_1", {}).get("rich_text", [])])
            elif block_type == "heading_2":
                text_to_add = "## " + "".join(
                    [t.get("plain_text", "") for t in block.get("heading_2", {}).get("rich_text", [])])
            elif block_type == "heading_3":
                text_to_add = "### " + "".join(
                    [t.get("plain_text", "") for t in block.get("heading_3", {}).get("rich_text", [])])
            elif block_type == "paragraph":
                text_to_add = "".join(
                    [t.get("plain_text", "") for t in block.get("paragraph", {}).get("rich_text", [])])
            elif block_type == "bulleted_list_item":
                text_to_add = "• " + "".join(
                    [t.get("plain_text", "") for t in block.get("bulleted_list_item", {}).get("rich_text", [])])
            elif block_type == "numbered_list_item":
                text_to_add = "1. " + "".join(
                    [t.get("plain_text", "") for t in block.get("numbered_list_item", {}).get("rich_text", [])])

            if text_to_add:
                text_content += text_to_add + "\n\n"

            if "children" in block:
                text_content += self.extract_text_content(block["children"])

        return text_content