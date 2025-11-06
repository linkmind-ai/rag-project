import traceback
import base64
import os
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage
from clients.llm_client import CustomLlmClient
from prompts.summary_prompt import SUMMARY_PROMPT

def img_to_b64(
        path: str = None,
        max_side: int = 1024,
) -> str:
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA"):
            bg = Image.new(RGB, im.size, (255,255,255))
            alpha = im.getchannel("A") if "A" in im.getbands() else None
            bg.paste(im, mask = alpha)
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size
        long_side = max(w, h)
        if long_side > max_side:
            scale = max_side / float(long_side)
            new_w, new_h = int(w * scale), int(h, scale)
            im = im.resize((new_w, new_h), Image.LANCZOS)

        ext = os.path.splitext(path)[1]

        if ext in (".png"):
            output_format = "PNG"
        elif ext in (".jpg", ".JPEG"):
            output_format = "JPEG"

        buf = BytesIO()

        if output_format.upper() == "JPEG":
            im.save(buf, format="JPEG", quality = 100, optimize = True, progressive = True)
            mime = "image/jpeg"
        elif output_format.upper() == "PNG":
            im.save(buf, format = "png", quality = 100)
            mime = "image/png"
        else:
            raise ValueError("Invalid Image Ext.")
        
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    

cli = CustomLlmClient(max_tokens = 32768)
llm = cli.get_llm()

req = """
주어진 이미지에 대해서 상세하게 설명해줘.
1. 이미지의 제목
2. 이미지에 대한 요약 (3문장 이내)
3. 이미지에 대한 설명 (3단락 이내)
"""
img_path = img_to_b64(
    path = "sample_img.png",
    max_side = 4096
)

msg = HumanMessage(content = [
    {"type": "text", "text": req},
    {"type": "image_url", "image_url": {"url": img_data}},
])

resp = llm.invoke([msg])
print(resp.content)