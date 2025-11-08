# jobs/batch/run_notion_etl.py

import config  # 루트의 config 패키지 임포트

# [수정] 'src' 패키지에서 클래스 임포트
from src.clients.notion_client import NotionClient
from src.common.text_processor import TextProcessor
from src.storage.elastic_store import ElasticStore


def run_etl():
    """
    전체 ETL 파이프라인 (Notion -> Text -> ES)을 실행합니다.
    """

    # 1. Extract (Notion)
    print("--- 1. EXTRACT: Fetching data from Notion... ---")
    notion = NotionClient(token=config.NOTION_TOKEN, version=config.NOTION_VERSION)
    page_id = input("Notion 페이지 ID를 입력하세요: ")
    page_id = page_id.replace("-", "")

    if not page_id:
        print("페이지 ID가 필요합니다. ETL을 중단합니다.")
        return

    all_blocks = notion.fetch_all_blocks(page_id)
    if not all_blocks:
        print("Notion에서 블록을 가져오지 못했습니다.")
        return

    text_content = notion.extract_text_content(all_blocks)

    try:
        with open(config.TXT_SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"✅ 텍스트 저장 완료: {config.TXT_SAVE_PATH}")
    except Exception as e:
        print(f"⚠️ 텍스트 파일 저장 실패: {e}")

    # 2. Transform (Text Processing)
    print("\n--- 2. TRANSFORM: Processing text (Chunking & Keywords)... ---")
    processor = TextProcessor(
        sbert_model_name=config.KEYBERT_MODEL  # config에서 모델 이름 주입
    )
    documents = processor.chunk_text(text_content, chunk_size=800)
    print(f"✅ 총 {len(documents)}개의 청크 생성 완료.")

    # 3. Load (Elasticsearch)
    print("\n--- 3. LOAD: Loading data into Elasticsearch... ---")
    store = ElasticStore(
        host=config.ES_HOST,
        api_id=config.ES_ID,
        api_key=config.ES_API_KEY,
        index_name=config.ES_INDEX_NAME,
        embedding_model_name=config.ES_EMBEDDING_MODEL  # config에서 모델 이름 주입
    )

    store.create_index_if_not_exists(dims=config.ES_EMBEDDING_DIMS)  # Dims 주입
    store.index_documents(documents)

    print("\n--- ✅ ETL Pipeline Completed Successfully! ---")


if __name__ == "__main__":
    run_etl()