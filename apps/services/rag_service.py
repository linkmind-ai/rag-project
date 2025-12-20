from rag_pipeline.notion_loader import get_page_content, fetch_all_blocks, extract_text_content
from rag_pipeline.text_chunker import TextChunker
from rag_pipeline.embed_indexer import embed_text
from rag_pipeline.retriever_reranker import hybrid_search

def run_rag_pipeline(page_id):
    page_data = get_page_content(page_id)
    all_blocks = fetch_all_blocks(page_id)
    text_content = extract_text_content(all_blocks)
    
    chunks = TextChunker.chunk_text_with_recursive_splitter(text_content)
    
    # 여기에 embed 저장, 검색, 리랭킹 순서로 실행
    ...
    return chunks