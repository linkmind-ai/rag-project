# src/clients/ollama_client.py

import ollama
import logging

logger = logging.getLogger(__name__)

class OllamaClient:
    """Ollama API와 통신하는 래퍼 클래스"""

    def __init__(self, host: str, model: str):
        self.model = model
        try:
            self.client = ollama.Client(host=host)
            self.client.list()
            logger.info(f"✅ Ollama 클라이언트 연결 성공 (Host: {host})")
        except Exception as e:
            logger.error(f"❌ Ollama 클라이언트 연결 실패 (Host: {host})", exc_info=True)
            raise

    def get_response(self, prompt: str) -> str | None:
        """프롬프트를 LLM에 전송하고 응답을 받습니다."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"❌ Ollama API 호출 오류: {e}", exc_info=True)
            return None