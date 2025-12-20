import re
from typing import List

from keybert import KeyBERT
from transformers import AutoModel
from kiwipiepy import Kiwi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from apps.models.documents import ChunkItem


class TextChunknizer:
    def __init__(self):
        self.kiwi = Kiwi()
        self.sbert_model = AutoModel.from_pretrained("jhgan/ko-sbert-sts")
        self.kw_model = KeyBERT(model=self.sbert_model)

    # 명사 추출
    def extract_nouns(self, text: str) -> List[str]:
        nouns = []
        for sent in self.kiwi.analyze(text):
            for tok in sent[0]:
                if tok.tag.startswith("NN"):
                    nouns.append(tok.form.strip())
        return nouns

    # ### HEADER ### 단위로 분리
    def split_text_with_headers(self, text: str, top_n=3, ngram_range=(1, 1)):
        chunks = re.split(r"### (.*?) ###", text)
        result = []

        for i in range(1, len(chunks), 2):
            title = chunks[i].strip()
            content = chunks[i + 1].strip() if i + 1 < len(chunks) else ""

            nouns = self.extract_nouns(content)
            nouns_text = " ".join(nouns)

            input_text = nouns_text if len(nouns) >= 3 else content

            keywords_tuple = self.kw_model.extract_keywords(
                input_text,
                keyphrase_ngram_range=ngram_range,
                top_n=top_n,
                use_mmr=True,
                diversity=0.3
            )
            keywords = "|".join([k for k, _ in keywords_tuple])

            # ⬇️ ChunkItem은 models에서 import
            result.append(ChunkItem(title, content, keywords))

        return result

    # 청크 분리 + header 메타데이터 유지
    def chunk_text_recursively(self, text: str, chunk_size=500, chunk_overlap=50):
        header_chunks = self.split_text_with_headers(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        final_chunks = []

        for item in header_chunks:
            temp_doc = Document(page_content=item.content)
            sub_chunks = splitter.split_documents([temp_doc])

            for ch in sub_chunks:
                meta = dict(ch.metadata)
                meta["title"] = item.title
                meta["keywords"] = item.keywords

                final_chunks.append(
                    Document(
                        page_content=ch.page_content,
                        metadata=meta
                    )
                )

        return final_chunks