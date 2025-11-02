# notion_loader.py

import requests
import json
import config

# Notion 인증 헤더
headers = {
    "Authorization": f"Bearer {config.NOTION_TOKEN}",
    "Notion-Version": config.NOTION_VERSION,
    "Content-Type": "application/json"
}


def get_page_content(page_id):
    """
    페이지 ID를 기반으로 페이지 속성 및 최상위 블록을 가져옵니다.
    """
    page_url = f"https://api.notion.com/v1/pages/{page_id}"
    response = requests.get(page_url, headers=headers)

    if response.status_code != 200:
        print(f"페이지 속성 가져오기 오류: {response.status_code}")
        print(response.text)
        return None

    page_data = response.json()

    # 페이지 내용(블록) 가져오기
    blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    response = requests.get(blocks_url, headers=headers)

    if response.status_code != 200:
        print(f"블록 가져오기 오류: {response.status_code}")
        print(response.text)
        return page_data  # 속성만이라도 반환

    blocks_data = response.json()
    page_data["content"] = blocks_data["results"]
    return page_data


def fetch_all_blocks(block_id):
    """
    블록의 모든 하위 블록을 재귀적으로 가져옵니다.
    """
    blocks_url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    response = requests.get(blocks_url, headers=headers)

    if response.status_code != 200:
        print(f"블록 가져오기 오류: {response.status_code}")
        print(response.text)
        return []

    blocks_data = response.json()["results"]

    has_children_types = [
        "paragraph", "bulleted_list_item", "numbered_list_item",
        "toggle", "child_page", "child_database", "column_list",
        "column", "table", "synced_block"
    ]

    all_blocks = []
    for block in blocks_data:
        all_blocks.append(block)

        if block.get("has_children") and block.get("type") in has_children_types:
            children = fetch_all_blocks(block["id"])
            if children:
                block["children"] = children

    return all_blocks


def extract_text_content(blocks):
    """
    블록 목록에서 텍스트 내용만 재귀적으로 추출합니다.
    (⚠️ 원본 노트북의 모든 타입을 여기에 포함해야 합니다)
    """
    text_content = ""
    for block in blocks:
        block_type = block.get("type")

        if block_type == "child_page":
            child_title = block.get("child_page", {}).get("title", "")
            if child_title:
                text_content += f"\n\n### {child_title} ###\n\n"

        elif block_type == "paragraph":
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

        if "children" in block:
            text_content += extract_text_content(block["children"])

    return text_content