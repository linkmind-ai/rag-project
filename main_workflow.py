# main_workflow.py

import json
import config
import notion_loader
import text_processor
import elasticsearch_manager


def run_notion_to_file():
    """Notion 페이지를 가져와 JSON과 TXT 파일로 저장합니다."""
    page_id = input("Notion 페이지 ID를 입력하세요: ")
    page_id = page_id.replace("-", "")

    print("페이지 내용을 가져오는 중...")
    page_data = notion_loader.get_page_content(page_id)
    if not page_data:
        print("페이지를 가져올 수 없습니다.")
        return None, None

    print("모든 블록을 재귀적으로 가져오는 중...")
    all_blocks = notion_loader.fetch_all_blocks(page_id)
    text_content = notion_loader.extract_text_content(all_blocks)

    # JSON 저장 (페이지 전체 데이터)
    try:
        with open(config.JSON_SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(page_data, f, ensure_ascii=False, indent=2)
        print(f"페이지 JSON 저장 완료: {config.JSON_SAVE_PATH}")
    except Exception as e:
        print(f"JSON 저장 오류: {e}")

    # TXT 저장 (추출된 텍스트만)
    try:
        with open(config.TXT_SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"페이지 TXT 저장 완료: {config.TXT_SAVE_PATH}")
    except Exception as e:
        print(f"TXT 저장 오류: {e}")

    return text_content


def run_process_and_index():
    """TXT 파일에서 텍스트를 로드하고, 청크로 분리한 뒤, Elasticsearch에 인덱싱합니다."""

    # 1. 텍스트 로드
    print(f"텍스트 파일 로드 중: {config.TXT_SAVE_PATH}")
    try:
        text = text_processor.load_text_from_file(config.TXT_SAVE_PATH)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다. 먼저 run_notion_to_file()을 실행하세요.")
        return
    except Exception as e:
        print(f"파일 로드 오류: {e}")
        return

    # 2. 청킹 실행
    print("텍스트 청킹 및 키워드 추출 중...")
    texts = text_processor.chunk_text_with_recursive_splitter(text, chunk_size=800)

    print(f"--- 총 {len(texts)}개의 청크 생성됨 ---")
    for i, chunk in enumerate(texts[:3]):  # 3개만 미리보기
        num_chars = len(chunk.page_content)
        print(f"[{i + 1}번째 청크] (글자 수: {num_chars}) (Title: {chunk.metadata.get('title')})")
        print(chunk.page_content[:100] + "...")

    # 3. Elasticsearch 연결
    es = elasticsearch_manager.get_es_client()
    if es is None:
        return

    tokenizer, model = elasticsearch_manager.get_embedding_model()
    if tokenizer is None or model is None:
        return

    # 4. 인덱스 생성
    elasticsearch_manager.create_es_index(es)

    # 5. (선택) 기존 문서 삭제
    # elasticsearch_manager.delete_all_docs(es)

    # 6. 문서 인덱싱
    elasticsearch_manager.index_documents(es, texts, tokenizer, model)

    # 7. 인덱싱 확인
    elasticsearch_manager.search_all_docs(es)