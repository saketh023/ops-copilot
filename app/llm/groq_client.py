import os
from openai import OpenAI


class GroqClient:
    def __init__(self):
        self.name = "groq"
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.model = "llama-3.1-8b-instant"

    def chat(self, messages, temperature=0.2, num_ctx=4096):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        return response.choices[0].message.content
