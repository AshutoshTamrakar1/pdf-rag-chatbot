from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    CHAT = "chat_message"
    CREATE_THREAD = "create_new_thread"
    LOAD_THREAD = "load_thread"
    DELETE_THREAD = "delete_thread"
    RENAME_THREAD = "rename_thread"
    GENERATE_MINDMAP = "generate_mindmap"
    GENERATE_PODCAST = "generate_podcast"
    FEEDBACK = "message_feedback"

class BaseRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    thread_id: str = Field(..., description="Thread ID")

class ChatRequest(BaseModel):
    session_id: str
    chat_session_id: Optional[str] = None
    user_input: str
    active_source_ids: List[str] = []
    # Optionally keep thread_id for backward compatibility, but always use chat_session_id
    thread_id: Optional[str] = None

    @validator("chat_session_id", always=True, pre=True)
    def populate_chat_session_id(cls, v, values):
        if v:
            return v
        return values.get("thread_id")

class ChatResponse(BaseModel):
    answer: str
    turn_id: str
    sources: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MindmapRequest(BaseRequest):
    source_id: str = Field(..., description="Source document ID")

class MindmapResponse(BaseModel):
    status: str
    markdown: str
    estimated_time: int
    thread_id: str
    source_id: str
    file_path: Optional[str] = None

class PodcastRequest(BaseRequest):
    source_id: str = Field(..., description="Source document ID")
    mindmap_id: Optional[str] = None

class PodcastResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    estimated_time: int
    thread_id: str
    source_id: str
    audio_path: Optional[str] = None

class WebSocketMessage(BaseModel):
    type: MessageType
    data: Dict[str, Any] = Field(default_factory=dict)
    client_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @validator('data')
    def validate_data(cls, v, values):
        msg_type = values.get('type')
        if msg_type == MessageType.CHAT:
            if not v.get('user_input'):
                raise ValueError("Chat messages must include user_input")
        elif msg_type == MessageType.GENERATE_MINDMAP:
            if not v.get('source_id'):
                raise ValueError("Mindmap generation requires source_id")
        return v

class WebSocketResponse(BaseModel):
    type: str
    data: Dict[str, Any]
    error: Optional[str] = None
    client_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# centralized logger available here for any model-related validation logging
from logging_config import get_logger
logger = get_logger(__name__)