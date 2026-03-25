"""
异步工具和函数支持 (Wave 2 T2.2 异步迁移)

提供异步兼容性支持，包括异步包装器、协程工具和性能监控。
"""

import asyncio
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncWrapper:
    """将同步函数包装为异步函数的简单工具类"""
    
    @staticmethod
    def to_async(func: Callable[..., T]) -> Callable[..., T]:
        """
        将同步函数包装为异步函数，在线程池中执行
        
        用于将现有的同步IO函数（如文件读写、网络请求）包装为异步版本
        """
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
        
        return async_wrapper
    
    @staticmethod
    async def run_parallel(tasks: List[Callable[[], T]], max_concurrent: int = 10) -> List[T]:
        """
        并行运行多个任务（替代 ThreadPoolExecutor）
        
        使用 asyncio.gather 和线程池执行同步函数
        """
        # 创建包装的异步任务
        loop = asyncio.get_event_loop()
        async_tasks = []
        
        for task in tasks:
            async_tasks.append(
                loop.run_in_executor(None, task)
            )
        
        # 分批执行以避免过度并发
        results = []
        for i in range(0, len(async_tasks), max_concurrent):
            batch = async_tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
        
        # 检查异常
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i} 执行失败: {result}")
                # 根据需要重新抛出异常或处理
                
        return results


class AsyncPerformanceMonitor:
    """异步性能监控器"""
    
    def __init__(self):
        self.measurements: Dict[str, Dict[str, Any]] = {}
    
    def measure(self, name: str):
        """
        异步性能测量装饰器
        
        用法:
            @monitor.measure("llm_call")
            async def async_llm_call():
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_cpu_time = time.process_time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    end_time = time.perf_counter()
                    end_cpu_time = time.process_time()
                    
                    elapsed = end_time - start_time
                    cpu_elapsed = end_cpu_time - start_cpu_time
                    
                    if name not in self.measurements:
                        self.measurements[name] = {
                            "count": 0,
                            "total_time": 0.0,
                            "total_cpu_time": 0.0,
                            "min_time": float('inf'),
                            "max_time": 0.0,
                        }
                    
                    self.measurements[name]["count"] += 1
                    self.measurements[name]["total_time"] += elapsed
                    self.measurements[name]["total_cpu_time"] += cpu_elapsed
                    self.measurements[name]["min_time"] = min(self.measurements[name]["min_time"], elapsed)
                    self.measurements[name]["max_time"] = max(self.measurements[name]["max_time"], elapsed)
                    
                    # 每10次记录一次性能数据
                    if self.measurements[name]["count"] % 10 == 0:
                        avg_time = self.measurements[name]["total_time"] / self.measurements[name]["count"]
                        avg_cpu_time = self.measurements[name]["total_cpu_time"] / self.measurements[name]["count"]
                        logger.info(
                            f"性能统计 [{name}]: "
                            f"调用={self.measurements[name]['count']}, "
                            f"平均时间={avg_time:.3f}s (min={self.measurements[name]['min_time']:.3f}s, max={self.measurements[name]['max_time']:.3f}s), "
                            f"平均CPU={avg_cpu_time:.3f}s"
                        )
                    
                    return result
                    
                except Exception as e:
                    end_time = time.perf_counter()
                    logger.error(f"执行失败 [{name}]: 耗时 {end_time - start_time:.3f}s, 错误: {e}")
                    raise
            
            return async_wrapper
        
        return decorator
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有性能统计"""
        stats = {}
        for name, data in self.measurements.items():
            if data["count"] > 0:
                stats[name] = {
                    "count": data["count"],
                    "total_time": data["total_time"],
                    "avg_time": data["total_time"] / data["count"],
                    "min_time": data["min_time"],
                    "max_time": data["max_time"],
                    "total_cpu_time": data["total_cpu_time"],
                    "avg_cpu_time": data["total_cpu_time"] / data["count"],
                }
        return stats


def sync_to_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    同步转异步的简便装饰器
    
    用法:
        @sync_to_async
        def sync_function(x, y):
            return x + y
        
        # 现在可以以异步方式调用
        result = await sync_function(1, 2)
    """
    return AsyncWrapper.to_async(func)


async def run_async_test():
    """测试异步基础设施是否正常工作"""
    monitor = AsyncPerformanceMonitor()
    
    @monitor.measure("test_task")
    async def test_task(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"完成延迟 {delay}s 的任务"
    
    # 并行运行多个测试任务
    tasks = [test_task(0.1) for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    logger.info(f"异步测试完成: {len(results)} 个任务")
    logger.info(f"性能统计: {monitor.get_stats()}")
    
    return results