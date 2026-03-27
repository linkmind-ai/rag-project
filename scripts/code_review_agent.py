import os
import json
import requests
import sys

def get_pr_details():
    event_path = os.getenv('GITHUB_EVENT_PATH')
    if not event_path or not os.path.exists(event_path):
        print("Not a GitHub PR event or GITHUB_EVENT_PATH missing.")
        return None, None
        
    with open(event_path, 'r', encoding='utf-8') as f:
        event = json.load(f)
        
    if 'pull_request' not in event:
        print("Not a pull_request event.")
        return None, None
        
    pr_number = event['pull_request']['number']
    repo = os.getenv('GITHUB_REPOSITORY')
    return repo, pr_number

def get_pr_diff(repo, pr_number, token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def analyze_code_with_llm(diff_text):
    prompt = f"""
당신은 시니어 개발자입니다. 아래 제공된 코드 변경 사항(git diff)을 주의 깊게 분석하여 다음 사항들을 포함한 코드 리뷰 리포트를 마크다운 형식으로 작성해주세요:
1. 변경 사항 요약 (핵심 의도)
2. 잠재적인 버그나 로직 오류
3. 보안 취약점 여부
4. 성능 및 구조적 개선 제안

[Git Diff 시작]
{diff_text}
[Git Diff 끝]
"""

    # Local Ollama or Groq API Configuration
    ollama_host = os.getenv('OLLAMA_HOST')
    ollama_model = os.getenv('OLLAMA_MODEL', 'hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:Q8_0')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    if ollama_host:
        print(f"Using Custom Ollama at {ollama_host} with model {ollama_model}")
        url = f"{ollama_host.rstrip('/')}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": "You are a helpful and experienced senior software engineer. Please answer in Korean."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()['message']['content']
        
    elif groq_api_key:
        print("Using Groq API")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192", # Default groq model or adjust as needed
            "messages": [
                {"role": "system", "content": "You are an expert code reviewer. Respond in Korean."},
                {"role": "user", "content": prompt}
            ]
        }
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
        
    else:
        raise ValueError("Neither OLLAMA_HOST nor GROQ_API_KEY environment variables are set. Please configure the LLM provider.")

def post_pr_comment(repo, pr_number, token, comment_body):
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "body": comment_body
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print("Comment posted successfully!")

def main():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("GITHUB_TOKEN is missing")
        sys.exit(1)
        
    repo, pr_number = get_pr_details()
    if not repo or not pr_number:
        print("Skipping code review, this script is currently designed to run on PR events.")
        sys.exit(0)
        
    print(f"Fetching diff for {repo} PR #{pr_number}...")
    try:
        diff_text = get_pr_diff(repo, pr_number, token)
        if not diff_text or len(diff_text.strip()) == 0:
            print("No visible changes to review.")
            sys.exit(0)
            
        print("Analyzing with LLM...")
        analysis_result = analyze_code_with_llm(diff_text)
        
        print("Posting result to GitHub PR...")
        review_comment = f"## 🤖 AI Code Review Agent (코드 리뷰 리포트)\n\n{analysis_result}"
        post_pr_comment(repo, pr_number, token, review_comment)
        
    except Exception as e:
        print(f"Error during code review: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
