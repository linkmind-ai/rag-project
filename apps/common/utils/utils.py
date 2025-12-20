import json

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)