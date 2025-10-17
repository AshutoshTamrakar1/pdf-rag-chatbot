import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Callable, Any
from config import get_settings
import functools
import asyncio

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "app.log")


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logger (console + rotating file). Call early in startup."""
    if log_file is None:
        log_file = DEFAULT_LOG_FILE

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for h in list(root.handlers):
        root.removeHandler(h)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Reduce noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def init_from_settings():
    """Initialize logging using Settings.LOG_LEVEL if present, otherwise INFO."""
    settings = get_settings()
    level = getattr(settings, "LOG_LEVEL", None)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if level is None:
        level = logging.INFO
    configure_logging(level=level)


def get_logger(name: str):
    return logging.getLogger(name)


def log_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to log exceptions raised inside functions.
    Works with sync and async functions.
    """
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception("Unhandled exception in %s: %s", func.__name__, e)
                raise
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception("Unhandled exception in %s: %s", func.__name__, e)
                raise
        return sync_wrapper