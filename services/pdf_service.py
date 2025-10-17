from fastapi import APIRouter, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from pathlib import Path
import shutil

from config import Settings, get_settings
from .exceptions import InvalidSessionError, ThreadNotFoundError
from .utils import validate_session, ensure_upload_dir
from db_manager import (
    add_source_to_chat_session,
    add_filename_to_uploaded_list,
    update_chat_session_field,
    get_chat_session_by_id
)
from auth import active_sessions
import uuid
from logging_config import get_logger, log_exceptions

logger = get_logger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF Operations"])

@router.post("/upload")
@log_exceptions
async def upload_pdf(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    chat_session_id: str = Form(...),
    settings: Settings = Depends(get_settings)
):
    """Handle PDF upload with improved error handling and logging"""
    logger.info(f"Processing PDF upload: {file.filename} for chat session {chat_session_id}")
    
    try:
        user_id = validate_session(session_id, active_sessions)
        chat_session_data = get_chat_session_by_id(chat_session_id, user_id)
        
        if not chat_session_data:
            raise ThreadNotFoundError()

        source_id = str(uuid.uuid4())
        upload_dir = ensure_upload_dir(
            user_id, chat_session_id, source_id, settings.UPLOAD_DIR
        )
        
        file_path = upload_dir / file.filename
        logger.debug(f"Saving file to: {file_path}")
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        new_source = {
            "source_id": source_id,
            "filename": file.filename,
            "filepath": str(file_path),
            "related_questions": [],
            "mindmap": {"path": None, "chat_messages": []},
            "podcast": {"data": None}
        }

        add_source_to_chat_session(chat_session_id, new_source)
        add_filename_to_uploaded_list(chat_session_id, file.filename)

        # Update chat session title if needed
        new_title = None
        if not chat_session_data.get('sources') and chat_session_data.get('title') == "New Chat":
            new_title = file.filename
            update_chat_session_field(chat_session_id, {'title': new_title})
            logger.info(f"Updated chat session title to: {new_title}")

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "new_source": new_source,
                "chat_session_id": chat_session_id,
                "new_title": new_title
            }
        )

    except Exception:
        # decorator logs full traceback
        raise HTTPException(status_code=500, detail="Internal server error during PDF upload")