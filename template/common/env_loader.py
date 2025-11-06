# config_loader.py
import json, os, sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"[ERROR] 환경변수 {name} 가 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)
    return v

def load_config(base_path="template/common/config.json"):
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    # 비밀은 env에서만 읽어서 주입
    base.setdefault("notion", {})["token"] = require("NOTION_TOKEN")
    base.setdefault("es", {})["api_key"] = require("ES_API_KEY")
    base["es"]["id"] = os.getenv("ES_ID", base["es"].get("id", ""))  # 선택적
    return base
