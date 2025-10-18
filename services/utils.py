import os
import json
from typing import Any, Dict, Optional
from pathlib import Path
from fastapi import HTTPException, status

# centralized logging + decorator import
from logging_config import get_logger, log_exceptions
logger = get_logger(__name__)

# If you have helper functions that can raise, add @log_exceptions above them.
# Example:
# @log_exceptions
# def some_helper(...):
#     ...existing code...

class RequestSimulator:
    """Utility class to simulate HTTP requests for service functions"""
    def __init__(self, json_data: Dict[str, Any]):
        self._json = json_data

    async def json(self) -> Dict[str, Any]:
        return self._json

def validate_session(session_id: str, active_sessions: Dict) -> str:
    """Validate session and return user_id"""
    if not session_id or session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid session"
        )
    return active_sessions[session_id]["user_id"]

def ensure_upload_dir(user_id: str, thread_id: str, source_id: str, base_dir: Path) -> Path:
    """Ensure upload directory exists and return path"""
    # Convert to Path if it's a string
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    upload_dir = base_dir / user_id / thread_id / source_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

async def handle_service_error(func: callable, *args, **kwargs) -> Dict[str, Any]:
    """Generic error handler for service functions"""
    try:
        return await func(*args, **kwargs)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Service error in {func.__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}"
        )