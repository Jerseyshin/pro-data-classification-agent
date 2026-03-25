import time
import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

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
    
    async def aprocess(self, state: ClassificationState) -> Dict[str, Any]:
        """
        异步处理状态，返回更新的状态字典
        
        默认实现：如果process已经是异步方法，直接调用；
        否则使用线程池执行同步版本，保持向后兼容
        """
        if inspect.iscoroutinefunction(self.process):
            # 如果子类已将process重写为异步方法，直接调用
            return await self.process(state)
        else:
            # 否则在线程池中执行同步版本，避免阻塞事件循环
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.process(state)
            )

    def __call__(self, state: ClassificationState) -> Dict[str, Any]:
        """LangGraph 同步调用入口，包裹 process 添加日志和计时"""
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
    
    async def __acall__(self, state: ClassificationState) -> Dict[str, Any]:
        """LangGraph 异步调用入口，包裹 aprocess 添加日志和计时"""
        self.logger.info("开始异步执行")
        start = time.perf_counter()
        try:
            result = await self.aprocess(state)
            elapsed = time.perf_counter() - start
            self.logger.info("异步执行完成，耗时 %.2fs", elapsed)
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.logger.error("异步执行失败，耗时 %.2fs，错误：%s", elapsed, e)
            raise
