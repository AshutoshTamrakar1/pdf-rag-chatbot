# This module previously duplicated logging setup — now it simply exposes the central helpers.
from logging_config import get_logger, log_exceptions, configure_logging, init_from_settings

# re-export for backward compatibility
__all__ = ["get_logger", "log_exceptions", "configure_logging", "init_from_settings"]

# Provide a module-level logger for callers that import services.logging
logger = get_logger(__name__)