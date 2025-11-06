import re
import os
import json
from keybert import KeyBERT
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from elasticsearch import Elasticsearch
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

# .env 로드
load_dotenv()

ES_HOST      = os.getenv("ES_HOST")
ES_ID        = os.getenv("ES_ID")          # 있으면 사용
ES_API_KEY   = os.getenv("ES_API_KEY")     # 있으면 사용
ES_INDEX     = os.getenv("ES_INDEX_NAME", "vector-test-index")
TXT_PATH     = os.getenv("TXT_PATH", "notion_page_content.txt")
CONTENTS_PATH = os.getenv("NOTION_CONTENTS_PATH")
TXT_PATH = os.getenv("NOTION_TXT_PATH", "notion_page_content.txt")
JSON_PATH = os.getenv("NOTION_JSON_PATH", "notion_page_content.json")

with open("template/common/config.json", "r") as f:
    cfg = json.load(f)

s_bert = SentenceTransformer(cfg['model']['keyword'])
kw_model = KeyBERT(model=s_bert)
kiwi = Kiwi()

# 텍스트 로드 함수
def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
    
def extract_nouns(text):
    nouns = []
    for sent in kiwi.analyze(text):
        for tok in sent[0]:
            if tok.tag.startswith("NN"):
                w = tok.form.strip()
                nouns.append(w)
    return nouns

# 상위 구분자 ("### 제목 ###") 기준으로 분할
def split_text_with_headers(text, top_n=3, ngram_range=(1, 1)):
    """
    "### 제목 ###" 패턴으로 문서를 나누고,
    각 chunk(content)에서 명사만 추출해 키워드(top_n) 반환.
    """
    chunks = re.split(r"### (.*?) ###", text)
    result = []

    for i in range(1, len(chunks), 2):
        title = chunks[i].strip()
        content = chunks[i + 1].strip() if i + 1 < len(chunks) else ""

        # 1) 명사만 추출하여 후보 텍스트 구성
        nouns = extract_nouns(content)
        nouns_text = " ".join(nouns)

        # 내용이 너무 짧거나 명사가 거의 없으면 원문으로 fallback
        input_text = nouns_text if len(nouns) >= 3 else content

        # 2) KeyBERT로 키워드 추출 (MMR로 다양성 확보)
        keywords_tuple = kw_model.extract_keywords(input_text, keyphrase_ngram_range=ngram_range, top_n=top_n)
        keywords = [k for k, _ in keywords_tuple]
        keywords = '|'.join(keywords)

        result.append({"title": title, "content": content, "keywords": keywords})

    return result

# 상위 구분자 기준 → 하위에서 RecursiveCharacterTextSplitter 적용
def chunk_text_with_recursive_splitter(text, chunk_size=500, chunk_overlap=50):
    header_chunks = split_text_with_headers(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = []

    for item in header_chunks:
        title = item.get("title", "")
        content = item.get("content", "")
        keywords = item.get("keywords", "")
        
        # 제목을 붙여서 RecursiveCharacterTextSplitter에 넘김
        temp_doc = Document(page_content=content)
        sub_chunks = text_splitter.split_documents([temp_doc])
        
        for sub_chunk in sub_chunks:
            # 제목을 메타데이터로 추가
            # 키워드도 같은 방식으로 추가 예정
            meta = dict(sub_chunk.metadata or {})
            meta["title"] = title
            meta["keywords"] = keywords
            
            final_chunks.append(Document(metadata=meta, page_content=sub_chunk.page_content))
    
    return final_chunks

# 텍스트 로드
text = load_text_from_file(CONTENTS_PATH + TXT_PATH)

# 청킹 실행
texts = chunk_text_with_recursive_splitter(text, chunk_size=800)

es = Elasticsearch(ES_HOST, api_key=(ES_ID, ES_API_KEY))

# 임베딩 모델 로드 (예: sentence-transformers)
tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["embedding"])
model = AutoModel.from_pretrained(cfg["model"]["embedding"])

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()  # 문장 임베딩 평균값
    return embeddings.numpy().tolist()

# 인덱스 생성 (dense_vector 타입 필드 포함)
index_name = ES_INDEX

if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        mappings={
            "properties": {
                "content": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": cfg["es"]["vec_dims"],
                    "index": True,
                    "similarity": "cosine",
                }
            }
        }
    )
    
# 인덱스 안에 문서 저장
for i, doc in enumerate(texts):
    content = doc.page_content
    metadata = doc.metadata
    
    body = {
        "content": content,
        "metadata": metadata,
        "embedding": embed_text(content),
    }
    es.index(index=index_name, id=f"chunk-{i}", document=body)

print("✅ 문서와 임베딩이 Elasticsearch에 저장되었습니다.")