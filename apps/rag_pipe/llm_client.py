# clients/llm_client.py

import ollama


class LLMClient:
    def __init__(self, host="https://ollama.nabee.ai.kr"):
        self.client = ollama.Client(host=host)
        self.model = "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M"

    def chat(self, messages):
        """
        messages =
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
        """
        response = self.client.chat(
            model=self.model,
            messages=messages
        )
        return response["message"]["content"]
