import requests
from common.config import NOTION_TOKEN, NOTION_VERSION
from common.utils import save_json

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json"
}

def get_page_content(page_id):
    page_url = f"https://api.notion.com/v1/pages/{page_id}"
    res = requests.get(page_url, headers=HEADERS)
    if res.status_code != 200:
        return None
    data = res.json()
    blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    blocks_res = requests.get(blocks_url, headers=HEADERS)
    if blocks_res.status_code == 200:
        data["content"] = blocks_res.json()["results"]
    return data

def fetch_all_blocks(block_id):
    """
    블록의 모든 하위 블록을 재귀적으로 가져옵니다.

    Args:
        block_id (str): 블록 ID

    Returns:
        list: 모든 하위 블록 목록
    """
    blocks_url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    response = requests.get(blocks_url, headers=headers)

    if response.status_code != 200:
        print(f"블록 가져오기 오류: {response.status_code}")
        print(response.text)
        return []

    blocks_data = response.json()["results"]

    # 하위 블록이 있는 블록 타입들
    has_children_types = [
        "paragraph", "bulleted_list_item", "numbered_list_item",
        "toggle", "child_page", "child_database", "column_list",
        "column", "table", "synced_block"
    ]

    all_blocks = []

    for block in blocks_data:
        all_blocks.append(block)

        # 하위 블록이 있는 경우 재귀적으로 가져오기
        if block.get("has_children") and block.get("type") in has_children_types:
            children = fetch_all_blocks(block["id"])
            # 하위 블록이 있는 경우에만 children 키 추가
            if children:
                block["children"] = children

    return all_blocks

def extract_text_content(blocks):
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

        # 자식 블록이 있으면 재귀적으로 처리
        if "children" in block:
            text_content += extract_text_content(block["children"])

    return text_content