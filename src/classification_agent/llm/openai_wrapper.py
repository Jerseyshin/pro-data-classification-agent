import json
from typing import Any, Dict, Optional
from openai import OpenAI, AsyncOpenAI

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """OpenAI LLM 包装"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Add instruction to output JSON
        prompt = prompt + "\n\nPlease respond with valid JSON only, no extra text."
        response_text = self.generate(prompt, temperature, max_tokens)

        # Clean up common issues
        response_text = response_text.strip()

        # Remove think tags that some models output (like Minimax)
        if "<think>" in response_text:
            # Extract everything after </think>
            parts = response_text.split("</think>")
            if len(parts) >= 2:
                response_text = parts[-1]
        # Remove any think closing tag that might be left
        response_text = response_text.replace("</think>", "").strip()

        # Remove markdown code block markers
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nResponse: {response_text}")

    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    async def agenerate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt = prompt + "\n\nPlease respond with valid JSON only, no extra text."
        response_text = await self.agenerate(prompt, temperature, max_tokens)

        response_text = response_text.strip()

        # Remove think tags that some models output (like Minimax)
        if "<think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) >= 2:
                response_text = parts[-1]
        response_text = response_text.replace("</think>", "").strip()

        # Remove markdown code block markers
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}\nResponse: {response_text}")
