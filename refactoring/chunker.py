import os
import re
from typing import List, Dict, Any
from keybert import KeyBERT
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch


# =========================
# 텍스트 로더
# =========================

class TextLoader:
    def __init__(self, base_path: str) -> None:
        self.base_path = base_path

    def load_text(self, file_name: str) -> str:
        file_path = os.path.join(self.base_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()


# =========================
# 키워드 추출 & 청킹 담당 클래스
# =========================

class NounExtractor:
    """텍스트에서 명사를 추출하는 인터페이스 역할 클래스."""
    def extract(self, text: str) -> List[str]:
        raise NotImplementedError


class KiwiNounExtractor(NounExtractor):
    """
    Kiwi로 명사를 추출하고,
    헤더 단위로 페이지를 나눠 각 페이지별 키워드를 할당하는 역할까지 담당.
    """
    def __init__(self, keyword_model_name: str) -> None:
        self.kiwi = Kiwi()
        s_bert = SentenceTransformer(keyword_model_name)
        self.kw_model = KeyBERT(model=s_bert)

    def extract(self, text: str) -> List[str]:
        nouns: List[str] = []
        for sent in self.kiwi.analyze(text):
            for tok in sent[0]:
                if tok.tag.startswith("NN"):
                    w = tok.form.strip()
                    nouns.append(w)
        return nouns

    def add_keywords_to_page(
        self,
        text: str,
        top_n: int = 3,
        ngram_range: tuple = (1, 1),
    ) -> List[Dict[str, Any]]:
        """
        \"### 제목 ###\" 패턴으로 문서를 나누고,
        각 chunk(content)에서 명사만 추출해 키워드(top_n)를 계산해
        title / content / keywords 형태로 반환.
        """
        chunks = re.split(r"### (.*?) ###", text)
        result: List[Dict[str, Any]] = []

        for i in range(1, len(chunks), 2):
            title = chunks[i].strip()
            content = chunks[i + 1].strip() if i + 1 < len(chunks) else ""

            # 1) 명사만 추출하여 후보 텍스트 구성
            nouns = self.extract(content)
            nouns_text = " ".join(nouns)

            # 내용이 너무 짧거나 명사가 거의 없으면 원문으로 fallback
            input_text = nouns_text if len(nouns) >= 3 else content

            # 2) KeyBERT로 키워드 추출
            keywords_tuple = self.kw_model.extract_keywords(
                input_text,
                keyphrase_ngram_range=ngram_range,
                top_n=top_n,
            )
            keywords = [k for k, _ in keywords_tuple]
            keywords = "|".join(keywords)

            result.append(
                {
                    "title": title,
                    "content": content,
                    "keywords": keywords,
                }
            )

        return result


class Chunker:
    """
    이미 헤더/키워드가 붙은 페이지 정보(header_chunks)를 받아
    RecursiveCharacterTextSplitter로 더 잘게 쪼개는 역할만 담당.
    """
    def __init__(self) -> None:
        pass

    def chunk_with_recursive_splitter(
        self,
        header_chunks: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 50) -> List[Document]:
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        final_chunks: List[Document] = []

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

                final_chunks.append(
                    Document(
                        metadata=meta,
                        page_content=sub_chunk.page_content,
                    )
                )

        return final_chunks



# =========================
# 임베딩 모델 래퍼
# =========================

class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text: str) -> List[float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # [batch, seq, hidden] → [batch, hidden]
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embeddings.cpu().numpy().tolist()