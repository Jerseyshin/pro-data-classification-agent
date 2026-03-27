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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Handle case where response is already a JSON string (some non-OpenAI compatible providers
            # that return the raw JSON string instead of parsed object
            if isinstance(response, str):
                # Parse JSON string to dict then extract content
                try:
                    if response.startswith("data:"):
                        response = response[5:]
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # 如果是字符串但不是JSON，可能是错误消息
                    return ""
            
            # Now response should be a dict or parsed object
            if isinstance(response, dict):
                # 防御性检查
                if not response.get("choices"):
                    return ""
                if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
                    return ""
                
                content = response["choices"][0].get("message", {}).get("content")
            else:
                # Normal OpenAI client parsed object
                # 防御性检查
                if not hasattr(response, 'choices'):
                    return ""
                if not response.choices or len(response.choices) == 0:
                    return ""
                    
                content = response.choices[0].message.content
            
            # 修复BUG: 正确处理None值
            if content is None:
                return ""
                
            # 处理思考标签（与generate_json保持一致）
            if "<think>" in content:
                parts = content.split("</think>")
                if len(parts) >= 2:
                    content = parts[-1]
            
            content = content.replace("</think>", "").strip()
            
            return content
        except Exception as e:
            # 简单返回空字符串而不是抛出异常
            return ""

    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Add instruction to output JSON
        prompt = prompt + "\n\nPlease respond with valid JSON only, no extra text."
        response_text = self.generate(prompt, temperature, max_tokens)

        # 如果响应为空，返回空JSON
        if not response_text or response_text.strip() == "":
            return {}

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

        # Find and extract the first code block with json if it exists
        if "```json" in response_text:
            # Extract content between ```json and next ```
            start = response_text.find("```json") + len("```json")
            end = response_text.find("```", start)
            if end > start:
                response_text = response_text[start:end]
        elif "```" in response_text:
            # Extract content between ``` and next ```
            start = response_text.find("```") + len("```")
            end = response_text.find("```", start)
            if end > start:
                response_text = response_text[start:end]

        # Clean up any remaining markers at edges
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # If there's text before the first {, skip everything before {
        if '{' in response_text:
            first_brace = response_text.index('{')
            if first_brace > 0:
                # If there's a closing }
                last_brace = response_text.rindex('}')
                if last_brace > first_brace:
                    response_text = response_text[first_brace:last_brace+1]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，返回空JSON而不是抛出异常
            # 这对于批量处理中的单个字段失败很重要
            return {}

    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Handle case where response is already a JSON string (some non-OpenAI compatible providers
            # that return the raw JSON string instead of parsed object
            if isinstance(response, str):
                # Parse JSON string to dict then extract content
                try:
                    if response.startswith("data:"):
                        response = response[5:]
                    response = json.loads(response)
                except json.JSONDecodeError:
                    # 如果是字符串但不是JSON，可能是错误消息
                    return ""
            
            # Now response should be a dict or parsed object
            if isinstance(response, dict):
                # 防御性检查
                if not response.get("choices"):
                    return ""
                if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
                    return ""
                
                content = response["choices"][0].get("message", {}).get("content")
            else:
                # Normal OpenAI client parsed object
                # 防御性检查
                if not hasattr(response, 'choices'):
                    return ""
                if not response.choices or len(response.choices) == 0:
                    return ""
                    
                content = response.choices[0].message.content
            
            # 修复BUG: 正确处理None值
            if content is None:
                return ""
                
            # 处理思考标签（与generate_json保持一致）
            if "<think>" in content:
                parts = content.split("</think>")
                if len(parts) >= 2:
                    content = parts[-1]
            
            content = content.replace("</think>", "").strip()
            
            return content
        except Exception as e:
            # 简单返回空字符串而不是抛出异常
            return ""

    async def agenerate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        prompt = prompt + "\n\nPlease respond with valid JSON only, no extra text."
        response_text = await self.agenerate(prompt, temperature, max_tokens)

        # 如果响应为空，返回空JSON
        if not response_text or response_text.strip() == "":
            return {}

        response_text = response_text.strip()

        # Remove think tags that some models output (like Minimax)
        if "<think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) >= 2:
                response_text = parts[-1]
        response_text = response_text.replace("</think>", "").strip()

        # Find and extract the first code block with json if it exists
        if "```json" in response_text:
            # Extract content between ```json and next ```
            start = response_text.find("```json") + len("```json")
            end = response_text.find("```", start)
            if end > start:
                response_text = response_text[start:end]
        elif "```" in response_text:
            # Extract content between ``` and next ```
            start = response_text.find("```") + len("```")
            end = response_text.find("```", start)
            if end > start:
                response_text = response_text[start:end]

        # Clean up any remaining markers at edges
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        # If there's text before the first {, skip everything before {
        if '{' in response_text:
            first_brace = response_text.index('{')
            if first_brace > 0:
                # If there's a closing }
                last_brace = response_text.rindex('}')
                if last_brace > first_brace:
                    response_text = response_text[first_brace:last_brace+1]

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # 如果JSON解析失败，返回空JSON而不是抛出异常
            return {}
