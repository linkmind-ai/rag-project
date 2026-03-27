from dotenv import load_dotenv
import os

from langchain_community.tools.tavily_search import TavilySearchResults

# .env 파일 로드
load_dotenv()

if __name__ == "__main__":
    # 환경변수 확인
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        print("TAVILY_API_KEY 가 .env 파일에 설정되어 있지 않습니다.")
    else:
        # Tavily tool 생성
        tool = TavilySearchResults(max_results=3, tavily_api_key=api_key)

        # 검색 실행
        results = tool.invoke("latest news about AI in 2025")
        web_results = "\n".join([d["content"] for d in results])

        print(web_results)
