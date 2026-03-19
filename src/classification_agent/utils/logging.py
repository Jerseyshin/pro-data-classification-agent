import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """获取 logger，首次调用时自动配置 StreamHandler 和格式"""
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        # 不向根 logger 传播，避免重复输出
        logger.propagate = False

    if level is not None:
        logger.setLevel(level)
    elif not logger.level:
        logger.setLevel(logging.INFO)

    return logger
