import os
import json
import requests
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경변수 불러오기
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
CONTENTS_PATH = os.getenv("NOTION_CONTENTS_PATH")
TXT_PATH = os.getenv("NOTION_TXT_PATH", "notion_page_content.txt")
JSON_PATH = os.getenv("NOTION_JSON_PATH", "notion_page_content.json")

# 환경변수 검증
if not NOTION_TOKEN:
    raise ValueError("❌ 환경변수 NOTION_TOKEN이 설정되어 있지 않습니다. .env 파일을 확인하세요.")

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
    "Content-Type": "application/json",
}

def get_page_content(page_id):
    """페이지 ID를 기반으로 페이지 내용을 가져옵니다."""
    page_url = f"https://api.notion.com/v1/pages/{page_id}"
    response = requests.get(page_url, headers=headers)
    if response.status_code != 200:
        print(f"오류 발생: {response.status_code}")
        print(response.text)
        return None

    page_data = response.json()

    blocks_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    response = requests.get(blocks_url, headers=headers)
    if response.status_code != 200:
        print(f"블록 가져오기 오류: {response.status_code}")
        print(response.text)
        return page_data

    blocks_data = response.json()
    page_data["content"] = blocks_data["results"]
    return page_data


def fetch_all_blocks(block_id):
    """블록의 모든 하위 블록을 재귀적으로 가져옵니다."""
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
    """블록에서 텍스트 내용만 추출."""
    text_content = ""
    for block in blocks:
        btype = block.get("type")
        if btype == "child_page":
            title = block.get("child_page", {}).get("title", "")
            if title:
                text_content += f"\n\n### {title} ###\n\n"
        elif btype == "paragraph":
            for t in block.get("paragraph", {}).get("rich_text", []):
                text_content += t.get("plain_text", "")
            text_content += "\n\n"
        elif btype in ["heading_1", "heading_2", "heading_3"]:
            level = btype.split("_")[1]
            for t in block[btype]["rich_text"]:
                text_content += "#" * int(level) + " " + t.get("plain_text", "")
            text_content += "\n\n"
        elif btype == "bulleted_list_item":
            for t in block[btype]["rich_text"]:
                text_content += "• " + t.get("plain_text", "")
            text_content += "\n"
        elif btype == "numbered_list_item":
            for t in block[btype]["rich_text"]:
                text_content += "1. " + t.get("plain_text", "")
            text_content += "\n"
        elif btype == "to_do":
            checked = block["to_do"].get("checked", False)
            for t in block["to_do"]["rich_text"]:
                text_content += f"[{'x' if checked else ' '}] " + t.get("plain_text", "")
            text_content += "\n"
        elif btype == "code":
            lang = block["code"].get("language", "")
            text_content += f"```{lang}\n"
            for t in block["code"]["rich_text"]:
                text_content += t.get("plain_text", "")
            text_content += "\n```\n\n"

        if "children" in block:
            text_content += extract_text_content(block["children"])
    return text_content


def main():
    page_id = input("Notion 페이지 ID를 입력하세요: ").replace("-", "")
    print("페이지 내용을 가져오는 중...")
    page_data = get_page_content(page_id)
    if not page_data:
        print("페이지를 가져올 수 없습니다.")
        return

    print("모든 블록을 재귀적으로 가져오는 중...")
    all_blocks = fetch_all_blocks(page_id)
    text_content = extract_text_content(all_blocks)

    # 저장 경로 생성
    os.makedirs(CONTENTS_PATH, exist_ok=True)
    json_file = os.path.join(CONTENTS_PATH, JSON_PATH)
    txt_file = os.path.join(CONTENTS_PATH, TXT_PATH)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=2)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"\n페이지 내용이 {JSON_PATH}과 {TXT_PATH}로 저장되었습니다.")


if __name__ == "__main__":
    main()