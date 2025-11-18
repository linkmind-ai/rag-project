import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


# =========================
# 환경 설정 클래스
# =========================

@dataclass
class NotionConfig:
    token: str
    version: str
    contents_path: str
    txt_path: str
    json_path: str

    @classmethod
    def from_env(cls) -> "NotionConfig":
        """환경 변수에서 설정을 로드."""
        load_dotenv()

        token = os.getenv("NOTION_TOKEN")
        if not token:
            raise ValueError("❌ 환경변수 NOTION_TOKEN이 설정되어 있지 않습니다. .env 파일을 확인하세요.")

        version = os.getenv("NOTION_VERSION", "2022-06-28")
        contents_path = os.getenv("NOTION_CONTENTS_PATH", ".")
        txt_path = os.getenv("NOTION_TXT_PATH", "notion_page_content.txt")
        json_path = os.getenv("NOTION_JSON_PATH", "notion_page_content.json")

        return cls(
            token=token,
            version=version,
            contents_path=contents_path,
            txt_path=txt_path,
            json_path=json_path,
        )


# =========================
# Notion API 클라이언트
# =========================

class NotionClient:
    BASE_URL = "https://api.notion.com/v1"

    def __init__(self, config: NotionConfig) -> None:
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Notion-Version": config.version,
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """공통 GET 요청 헬퍼."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            print(f"요청 오류: {response.status_code}")
            print(response.text)
            return None
        return response.json()

    def get_page_with_top_blocks(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        페이지 메타데이터 + 최상위 블록(children)을 함께 가져온다.
        (기존 get_page_content 함수 역할)
        """
        page_data = self._get(f"/pages/{page_id}")
        if page_data is None:
            return None

        blocks_data = self._get(f"/blocks/{page_id}/children")
        if blocks_data is None:
            return page_data

        page_data["content"] = blocks_data.get("results", [])
        return page_data

    def fetch_all_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """
        블록의 모든 하위 블록을 재귀적으로 가져온다.
        (기존 fetch_all_blocks 함수 역할)
        """
        blocks_data = self._get(f"/blocks/{block_id}/children")
        if blocks_data is None:
            return []

        blocks = blocks_data.get("results", [])

        has_children_types = [
            "paragraph", "bulleted_list_item", "numbered_list_item",
            "toggle", "child_page", "child_database", "column_list",
            "column", "table", "synced_block",
        ]

        all_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            all_blocks.append(block)
            if block.get("has_children") and block.get("type") in has_children_types:
                children = self.fetch_all_blocks(block["id"])
                if children:
                    block["children"] = children
        return all_blocks


# =========================
# 텍스트 추출기
# =========================

class NotionContentExtractor:
    """Notion 블록 구조에서 텍스트를 추출하는 책임을 가지는 클래스."""

    @classmethod
    def extract_text_content(cls, blocks: List[Dict[str, Any]]) -> str:
        """블록에서 텍스트 내용만 추출."""
        text_content = ""
        for block in blocks:
            btype = block.get("type")

            # child_page 제목
            if btype == "child_page":
                title = block.get("child_page", {}).get("title", "")
                if title:
                    text_content += f"\n\n### {title} ###\n\n"

            # 일반 문단
            elif btype == "paragraph":
                for t in block.get("paragraph", {}).get("rich_text", []):
                    text_content += t.get("plain_text", "")
                text_content += "\n\n"

            # 헤딩
            elif btype in ["heading_1", "heading_2", "heading_3"]:
                level = btype.split("_")[1]
                for t in block[btype].get("rich_text", []):
                    text_content += "#" * int(level) + " " + t.get("plain_text", "")
                text_content += "\n\n"

            # 불릿 리스트
            elif btype == "bulleted_list_item":
                for t in block[btype].get("rich_text", []):
                    text_content += "• " + t.get("plain_text", "")
                text_content += "\n"

            # 넘버드 리스트
            elif btype == "numbered_list_item":
                for t in block[btype].get("rich_text", []):
                    text_content += "1. " + t.get("plain_text", "")
                text_content += "\n"

            # To-do 리스트
            elif btype == "to_do":
                checked = block["to_do"].get("checked", False)
                for t in block["to_do"].get("rich_text", []):
                    text_content += f"[{'x' if checked else ' '}] " + t.get("plain_text", "")
                text_content += "\n"

            # 코드 블록
            elif btype == "code":
                lang = block["code"].get("language", "")
                text_content += f"```{lang}\n"
                for t in block["code"].get("rich_text", []):
                    text_content += t.get("plain_text", "")
                text_content += "\n```\n\n"

            # 자식 블록 재귀 처리
            if "children" in block:
                text_content += cls.extract_text_content(block["children"])

        return text_content


# =========================
# 전체 흐름을 담당하는 클래스
# =========================

class NotionPageExporter:
    def __init__(self, client: NotionClient, config: NotionConfig) -> None:
        self.client = client
        self.config = config

    def export_page(self, page_id: str) -> None:
        """페이지 내용을 가져와 JSON + TXT 파일로 저장."""
        cleaned_page_id = page_id.replace("-", "")
        print("페이지 내용을 가져오는 중...")
        page_data = self.client.get_page_with_top_blocks(cleaned_page_id)
        if not page_data:
            print("페이지를 가져올 수 없습니다.")
            return

        print("모든 블록을 재귀적으로 가져오는 중...")
        all_blocks = self.client.fetch_all_blocks(cleaned_page_id)
        text_content = NotionContentExtractor.extract_text_content(all_blocks)

        # 저장 경로 생성
        os.makedirs(self.config.contents_path, exist_ok=True)
        json_file = os.path.join(self.config.contents_path, self.config.json_path)
        txt_file = os.path.join(self.config.contents_path, self.config.txt_path)

        # JSON: 원본 page_data 그대로 저장 (기존 코드와 동일)
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)

        # TXT: 재귀적으로 수집한 text_content 저장
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text_content)

        print(f"\n페이지 내용이 {self.config.json_path}과 {self.config.txt_path}로 저장되었습니다.")


# =========================
# CLI 엔트리 포인트
# =========================

def main():
    config = NotionConfig.from_env()
    client = NotionClient(config)
    exporter = NotionPageExporter(client, config)

    page_id = input("Notion 페이지 ID를 입력하세요: ")
    exporter.export_page(page_id)


if __name__ == "__main__":
    main()