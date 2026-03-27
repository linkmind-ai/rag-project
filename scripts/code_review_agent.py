import os
import json
import requests
import sys

def get_event_details():
    event_path = os.getenv('GITHUB_EVENT_PATH')
    event_name = os.getenv('GITHUB_EVENT_NAME')
    repo = os.getenv('GITHUB_REPOSITORY')
    
    if not event_path or not os.path.exists(event_path):
        print("GITHUB_EVENT_PATH missing.")
        return None, None, None, None
        
    with open(event_path, 'r', encoding='utf-8') as f:
        event = json.load(f)
        
    if event_name == 'pull_request':
        return 'pull_request', repo, event['pull_request']['number'], None
    elif event_name == 'push':
        before = event.get('before')
        after = event.get('after')
        return 'push', repo, before, after
    else:
        return event_name, repo, None, None

def filter_large_machine_files(diff_text):
    """uv.lock 등 기계가 생성하는 거대 파일의 Diff를 LLM 분석에서 제외합니다."""
    if not diff_text:
        return diff_text
        
    blocks = diff_text.split("diff --git ")
    filtered_blocks = []
    for block in blocks:
        if not block.strip():
            continue
        first_line = block.split('\n', 1)[0]
        # uv.lock, package-lock.json, poetry.lock, requirements 등은 리뷰 생략
        if any(ignored in first_line for ignored in ["uv.lock", ".lock", "requirements"]):
            print(f"Skipping large/machine-generated file in diff: {first_line}")
            continue
            
        filtered_blocks.append("diff --git " + block)
    return "".join(filtered_blocks)

def get_pr_diff(repo, pr_number, token):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return filter_large_machine_files(response.text)

def get_push_diff(repo, before, after, token):
    url = f"https://api.github.com/repos/{repo}/compare/{before}...{after}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3.diff"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return filter_large_machine_files(response.text)

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

    print(f"[Debug] Diff length: {len(diff_text)} characters")
    
    ollama_host = os.getenv('OLLAMA_HOST')
    # fallback to 1.2B if not specified to avoid loading massive 32B model, but keeping user default if set
    ollama_model = os.getenv('OLLAMA_MODEL')
    if not ollama_model:
        ollama_model = 'hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:Q8_0'
        
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # helper for Groq fallback
    def call_groq():
        if not groq_api_key:
            raise ValueError("Ollama failed and GROQ_API_KEY is not set for fallback.")
        print("Using Groq API...")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are an expert code reviewer. Respond in Korean."},
                {"role": "user", "content": prompt}
            ]
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    
    if ollama_host:
        print(f"Using Custom Ollama at {ollama_host} with model {ollama_model}")
        cf_client_id = os.getenv('CF_ACCESS_CLIENT_ID')
        cf_client_secret = os.getenv('CF_ACCESS_CLIENT_SECRET')
        url = f"{ollama_host.rstrip('/')}/api/chat"
        headers = {}
        if cf_client_id and cf_client_secret:
            headers['CF-Access-Client-Id'] = cf_client_id
            headers['CF-Access-Client-Secret'] = cf_client_secret
        
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": "You are a helpful and experienced senior software engineer. Please answer in Korean."},
                {"role": "user", "content": prompt}
            ],
            "stream": True # 스트림 연결을 유지해 Cloudflare 524 타임아웃 방지
        }
        
        try:
            # 타임아웃 180초 (사용자 요청)
            resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=180)
            resp.raise_for_status()
            
            full_content = ""
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        full_content += chunk["message"]["content"]
            if not full_content.strip():
                raise ValueError("Ollama returned an empty response.")
            return full_content
            
        except Exception as e:
            print(f"Ollama Request Failed: {e}")
            print("Falling back to Groq API due to Ollama timeout/error...")
            return call_groq()
            
    elif groq_api_key:
        return call_groq()
        
    else:
        raise ValueError("Neither OLLAMA_HOST nor GROQ_API_KEY are set.")

def post_pr_comment(repo, pr_number, token, comment_body):
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {"body": comment_body}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print("PR Comment posted successfully!")

def post_commit_comment(repo, commit_sha, token, comment_body):
    url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {"body": comment_body}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    print("Commit Comment posted successfully!")

def main():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("GITHUB_TOKEN is missing")
        sys.exit(1)
        
    event_name, repo, arg1, arg2 = get_event_details()
    if not event_name or not repo:
        print("Could not determine event details.")
        sys.exit(0)
        
    try:
        if event_name == 'pull_request':
            pr_number = arg1
            print(f"Fetching diff for PR #{pr_number}...")
            diff_text = get_pr_diff(repo, pr_number, token)
        elif event_name == 'push':
            before, after = arg1, arg2
            if not before or not after or before.replace('0', '') == '':
                # Using the single commit diff if before is 000000 or missing
                diff_text = get_push_diff(repo, after + "^", after, token)
            else:
                print(f"Fetching diff for push {before}...{after}...")
                diff_text = get_push_diff(repo, before, after, token)
        else:
            print(f"Skipping code review for event: {event_name}")
            sys.exit(0)
            
        if not diff_text or len(diff_text.strip()) == 0:
            print("No visible changes to review.")
            sys.exit(0)
            
        print("Analyzing with LLM...")
        analysis_result = analyze_code_with_llm(diff_text)
        
        review_comment = f"## 🤖 AI Code Review Agent (코드 리뷰 리포트)\n\n{analysis_result}"
        
        print("Posting result to GitHub...")
        if event_name == 'pull_request':
            post_pr_comment(repo, arg1, token, review_comment)
        elif event_name == 'push':
            post_commit_comment(repo, arg2, token, review_comment)
            
    except Exception as e:
        print(f"Error during code review: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
