from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
from datetime import datetime

from config import Settings, get_settings
from .models import ChatRequest, ChatResponse
from .exceptions import InvalidSessionError, ThreadNotFoundError
from .utils import validate_session, handle_service_error
from db_manager import (
    add_turn_to_general_chat,
    add_turn_to_multi_source_chat,
    add_question_to_source,
    get_chat_session_by_id
)
from ai_engine import (
    chat_completion_LlamaModel_ws,
    chat_completion_Gemma_ws,
    chat_completion_with_pdf_ws,
    chat_completion_with_multiple_pdfs_ws,
    HISTORY_LENGTH
)
from auth import active_sessions
import uuid
from logging_config import get_logger, log_exceptions

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat Operations"])

def process_chat_completion(
    user_input: str,
    chat_session_data: Dict,
    active_source_ids: List[str],
    settings: Settings,
    model: Optional[str] = None
) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    """Select correct ai_engine async-generator based on context and requested model.

    - If one or more PDFs selected -> use RAG functions (llama3 RAG).
    - If no PDFs selected -> use Gemma if model == "gemma", otherwise llama3.
    Returns an async generator yielding (chunk, error).
    """
    logger.debug(f"Processing chat with {len(active_source_ids)} active sources; model={model}")

    history: List[Dict[str, str]] = chat_session_data.get("messages", []) or []

    if len(active_source_ids) == 1:
        source_data = next(
            (s for s in chat_session_data.get("sources", [])
             if s.get("source_id") == active_source_ids[0]),
            None
        )
        if not source_data or not source_data.get("filepath"):
            async def _err():
                yield None, "Source document not found"
            return _err()

        # single-PDF RAG uses llama3-based RAG
        return chat_completion_with_pdf_ws(
            user_input,
            history,
            source_data["filepath"]
        )
    elif len(active_source_ids) > 1:
        pdf_paths = [
            s.get("filepath") for s in chat_session_data.get("sources", [])
            if s.get("source_id") in active_source_ids and s.get("filepath")
        ]
        return chat_completion_with_multiple_pdfs_ws(
            user_input,
            history,
            pdf_paths
        )
    else:
        # general chat: allow switching between llama3 and gemma
        if model and model.lower() == "gemma":
            return chat_completion_Gemma_ws(user_input, history)
        return chat_completion_LlamaModel_ws(user_input, history)

@router.post("/send", response_model=ChatResponse)
@log_exceptions
async def send_chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    model: Optional[str] = None
) -> ChatResponse:
    """Handle chat messages. Optional query param `model` can be 'gemma' to use Gemma for general chat."""
    logger.info(f"Processing chat request for chat session: {request.chat_session_id}; model={model}")

    try:
        user_id = validate_session(request.session_id, active_sessions)
        chat_session_data = get_chat_session_by_id(request.chat_session_id, user_id)

        if not chat_session_data:
            raise ThreadNotFoundError()

        # get async generator (do NOT await)
        completion_generator = process_chat_completion(
            request.user_input,
            chat_session_data,
            request.active_source_ids,
            settings,
            model=model
        )

        # Stream / collect response
        full_response = ""
        async for chunk, error in completion_generator:
            if error:
                logger.error(f"Error in chat completion: {error}")
                raise HTTPException(status_code=500, detail=error)
            if chunk:
                full_response += chunk

        # Save chat turn
        turn_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        new_turn = {
            "id": turn_id,
            "user_query": request.user_input,
            "assistant_response": full_response,
            "timestamp": timestamp,
            "feedback": None
        }

        # Save based on chat type
        if len(request.active_source_ids) > 1:
            add_turn_to_multi_source_chat(
                request.chat_session_id,
                request.active_source_ids,
                new_turn
            )
        elif len(request.active_source_ids) == 1:
            add_question_to_source(
                request.chat_session_id,
                request.active_source_ids[0],
                new_turn
            )
        else:
            add_turn_to_general_chat(
                request.chat_session_id,
                new_turn
            )

        logger.info(f"Chat turn saved with ID: {turn_id}")
        return ChatResponse(
            answer=full_response,
            turn_id=turn_id,
            sources=request.active_source_ids
        )

    except Exception:
        # decorator already logs exception; re-raise to let FastAPI handle response
        raise