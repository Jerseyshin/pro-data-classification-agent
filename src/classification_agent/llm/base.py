from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseLLM(ABC):
    """LLM 抽象基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """生成文本回答"""
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """生成JSON格式回答并解析"""
        pass

    async def agenerate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """异步生成文本回答"""
        raise NotImplementedError("Async generation not implemented")

    async def agenerate_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """异步生成JSON格式回答并解析"""
        raise NotImplementedError("Async JSON generation not implemented")
