# src/common/text_processor.py

import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from keybert import KeyBERT
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer


class TextProcessor:
    """텍스트를 청크로 분할하고 키워드를 추출하는 클래스"""

    def __init__(self, sbert_model_name: str):
        self.sbert_model_name = sbert_model_name
        self.sbert = None
        self.kw_model = None
        self.okt = None

    def _get_okt(self):
        if self.okt is None:
            print("[Lazy Load] Initializing Konlpy Okt model...")
            self.okt = Okt()
        return self.okt

    def _get_kw_model(self):
        if self.kw_model is None:
            print(f"[Lazy Load] Initializing KeyBERT model: {self.sbert_model_name}...")
            self.sbert = SentenceTransformer(self.sbert_model_name)
            self.kw_model = KeyBERT(model=self.sbert)
        return self.kw_model

    def _extract_nouns(self, text: str) -> list:
        local_okt = self._get_okt()
        return local_okt.nouns(text)

    def _split_text_with_headers(self, text: str, top_n=3) -> list:
        chunks = re.split(r"### (.*?) ###", text)
        result = []
        local_kw_model = self._get_kw_model()

        for i in range(1, len(chunks), 2):
            title = chunks[i].strip()
            content = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
            keywords = ""

            if content.strip():
                nouns = self._extract_nouns(content)
                input_text = " ".join(nouns) if len(nouns) >= 3 else content

                if input_text.strip():
                    try:
                        keywords_tuple = local_kw_model.extract_keywords(
                            input_text, keyphrase_ngram_range=(1, 1), top_n=top_n
                        )
                        keywords = '|'.join([k for k, _ in keywords_tuple])
                    except Exception as e:
                        print(f"KeyBERT 추출 오류: {e}")
                        keywords = ""

            result.append({"title": title, "content": content, "keywords": keywords})
        return result

    def chunk_text(self, text: str, chunk_size=800, chunk_overlap=50) -> list[Document]:
        """최종적으로 텍스트를 Langchain Document 청크 리스트로 반환"""
        header_chunks = self._split_text_with_headers(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        final_chunks = []

        for item in header_chunks:
            temp_doc = Document(page_content=item.get("content", ""))
            sub_chunks = text_splitter.split_documents([temp_doc])

            for sub_chunk in sub_chunks:
                meta = {
                    "title": item.get("title", ""),
                    "keywords": item.get("keywords", "")
                }
                final_chunks.append(Document(metadata=meta, page_content=sub_chunk.page_content))

        return final_chunks