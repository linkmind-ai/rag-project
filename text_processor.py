# text_processor.py

import re
import config
from keybert import KeyBERT
# from kiwipiepy import Kiwi # 제거
from konlpy.tag import Okt  # 추가
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 모델 변수를 None으로 초기화 ---
sbert = None
kw_model = None
# kiwi = None # 제거
okt = None  # 추가


def _get_okt():
    """Konlpy Okt 모델을 지연 로딩합니다."""
    global okt
    if okt is None:
        print("\n[Lazy Load] Initializing Konlpy Okt model...")
        # Okt는 처음 실행 시 Java(JPype)를 로드합니다.
        okt = Okt()
    return okt


def _get_kw_model():
    """KeyBERT 및 SBERT 모델을 지연 로딩합니다."""
    global sbert, kw_model
    if kw_model is None:
        print(f"\n[Lazy Load] Initializing KeyBERT model: {config.KEYBERT_MODEL}...")
        sbert = SentenceTransformer(config.KEYBERT_MODEL)
        kw_model = KeyBERT(model=sbert)
    return kw_model


# --- 함수 정의 ---

def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_nouns(text):
    # --- Kiwi를 Okt로 교체 ---
    local_okt = _get_okt()
    if local_okt is None:
        return []

    # Okt는 .nouns() 메서드로 명사만 바로 추출할 수 있습니다.
    nouns = local_okt.nouns(text)
    return nouns


def split_text_with_headers(text, top_n=3, ngram_range=(1, 1)):
    """
    "### 제목 ###" 패턴으로 문서를 나누고,
    각 chunk(content)에서 명사만 추출해 키워드(top_n) 반환.
    """
    chunks = re.split(r"### (.*?) ###", text)
    result = []

    local_kw_model = _get_kw_model()

    if local_kw_model is None:
        print("KeyBERT 모델이 로드되지 않았습니다. 키워드 추출을 건너뜁니다.")

    for i in range(1, len(chunks), 2):
        title = chunks[i].strip()
        content = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
        keywords = ""

        if local_kw_model and content.strip():
            # extract_nouns()는 이제 Okt를 사용합니다.
            nouns = extract_nouns(content)
            nouns_text = " ".join(nouns)
            input_text = nouns_text if len(nouns) >= 3 else content

            if input_text.strip():
                try:
                    keywords_tuple = local_kw_model.extract_keywords(
                        input_text, keyphrase_ngram_range=ngram_range, top_n=top_n
                    )
                    keywords = [k for k, _ in keywords_tuple]
                    keywords = '|'.join(keywords)
                except Exception as e:
                    print(f"KeyBERT 키워드 추출 오류: {e}")
                    keywords = ""

        result.append({"title": title, "content": content, "keywords": keywords})
    return result


def chunk_text_with_recursive_splitter(text, chunk_size=500, chunk_overlap=50):
    header_chunks = split_text_with_headers(text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    final_chunks = []

    for item in header_chunks:
        title = item.get("title", "")
        content = item.get("content", "")
        keywords = item.get("keywords", "")

        temp_doc = Document(page_content=content)
        sub_chunks = text_splitter.split_documents([temp_doc])

        for sub_chunk in sub_chunks:
            meta = dict(sub_chunk.metadata or {})
            meta["title"] = title
            meta["keywords"] = keywords

            final_chunks.append(Document(metadata=meta, page_content=sub_chunk.page_content))

    return final_chunks