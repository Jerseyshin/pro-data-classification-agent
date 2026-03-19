import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from classification_agent.graph.state import ClassificationState
from classification_agent.llm.base import BaseLLM
from classification_agent.utils.logging import get_logger


class BaseNode(ABC):
    """节点基类：统一处理日志、计时"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def process(self, state: ClassificationState) -> Dict[str, Any]:
        """处理状态，返回更新的状态字典"""
        pass

    def __call__(self, state: ClassificationState) -> Dict[str, Any]:
        """LangGraph 调用入口，包裹 process 添加日志和计时"""
        self.logger.info("开始执行")
        start = time.perf_counter()
        try:
            result = self.process(state)
            elapsed = time.perf_counter() - start
            self.logger.info("执行完成，耗时 %.2fs", elapsed)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.logger.error("执行失败，耗时 %.2fs，错误：%s", elapsed, e)
            raise
