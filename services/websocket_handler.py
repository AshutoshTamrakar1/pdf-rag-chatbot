from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any, Optional, List
from logging_config import get_logger, log_exceptions
import json
import asyncio
from datetime import datetime
import uuid

from config import Settings, get_settings
from .models import (
    ChatRequest, 
    MindmapRequest, 
    WebSocketMessage, 
    WebSocketResponse
)
from .exceptions import InvalidSessionError
from .utils import validate_session
from .chat_service import send_chat
from .mindmap_service import generate_mindmap, generate_podcast
from db_manager import (
    get_user_chat_session_list,
    get_chat_session_by_id,
    create_chat_session_in_db,
    mark_chat_session_as_deleted,
    rename_chat_session_title,
    COLLECTION_CHAT_SESSIONS
)
from auth import active_sessions
from ai_engine import (
    chat_completion_with_pdf_ws,
    chat_completion_with_multiple_pdfs_ws,
    chat_completion_LlamaModel_ws,
    chat_completion_Gemma_ws
)

logger = get_logger(__name__)
router = APIRouter(tags=["WebSocket"])

class ConnectionManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        self._connections: Dict[str, WebSocket] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._settings: Optional[Settings] = None

    async def connect(
        self, 
        websocket: WebSocket, 
        session_id: str,
        settings: Settings
    ) -> None:
        """Handle new WebSocket connection"""
        await websocket.accept()
        self._connections[session_id] = websocket
        self._settings = settings
        logger.info(f"New WebSocket connection: {session_id}")

    def disconnect(self, session_id: str) -> None:
        """Handle WebSocket disconnection"""
        if session_id in self._connections:
            del self._connections[session_id]
        if session_id in self._tasks:
            self._tasks[session_id].cancel()
            del self._tasks[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_json(
        self, 
        session_id: str, 
        data: Dict[str, Any]
    ) -> None:
        """Send JSON response to client"""
        if session_id in self._connections:
            await self._connections[session_id].send_json(data)

    async def handle_message(
        self,
        session_id: str,
        message: WebSocketMessage
    ) -> None:
        """Route and handle incoming WebSocket messages"""
        try:
            if message.type == "create_new_thread":
                await self._handle_create_thread(session_id)
            elif message.type == "chat_message":
                await self._handle_chat(session_id, message)
            elif message.type == "generate_mindmap":
                await self._handle_mindmap(session_id, message)
            elif message.type == "generate_podcast":
                await self._handle_podcast(session_id, message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                
        except Exception:
            # decorator/context logging already records traceback; send minimal error to client
            logger.exception("Error handling message")
            await self.send_json(session_id, {"type": "error", "error": "Internal server error"})

    async def _handle_create_thread(self, session_id: str) -> None:
        """Handle thread creation request"""
        user_id = validate_session(session_id, active_sessions)
        chat_session_id = str(uuid.uuid4())
        
        new_chat_session = {
            "id": chat_session_id,
            "user_id": user_id,
            "title": "New Chat",
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "sources": []
        }
        
        create_chat_session_in_db(new_chat_session)
        await self.send_json(session_id, {
            "type": "thread_created",
            "chat_session_id": chat_session_id,
            "title": "New Chat"
        })
        logger.info(f"Created new chat session: {chat_session_id}")

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    settings: Settings = Depends(get_settings)
):
    """Main WebSocket endpoint"""
    try:
        # Validate session and connect
        if not session_id or session_id not in active_sessions:
            await websocket.close(code=1008)
            return

        await manager.connect(websocket, session_id, settings)
        user_id = validate_session(session_id, active_sessions)

        # Send initial chat session list
        chat_sessions = get_user_chat_session_list(user_id)
        await manager.send_json(session_id, {
            "type": "history_list",
            "chat_sessions": chat_sessions
        })

        # Main message handling loop
        while True:
            data = await websocket.receive_json()
            message = WebSocketMessage(**data)
            await manager.handle_message(session_id, message)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        manager.disconnect(session_id)