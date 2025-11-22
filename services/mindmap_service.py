# """
# All mindmap and podcast endpoints and logic are disabled as per user request.
# This file is intentionally left with no active endpoints.
# """

# from fastapi import APIRouter, Depends, HTTPException, status
# from typing import Dict, Any, Optional
# import logging
# from pathlib import Path
# from datetime import datetime

# from config import Settings, get_settings
# from .models import MindmapRequest, MindmapResponse, PodcastRequest, PodcastResponse
# from .exceptions import InvalidSessionError, ThreadNotFoundError, SourceNotFoundError
# from .utils import validate_session, handle_service_error
# from db_manager import (
#     update_source_field,
#     get_chat_session_by_id
# )
# from ai_engine import (
#     generate_mindmap_from_pdf,
#     estimate_mindmap_generation_time,
#     generate_podcast_from_mindmap,
#     estimate_podcast_generation_time
# )
# from auth import active_sessions
# from logging_config import get_logger, log_exceptions

# logger = get_logger(__name__)
# router = APIRouter(tags=["Mindmap & Podcast Operations"])

# """ Disabled mindmap and podcast endpoints as per user request
# # @router.post("/", response_model=MindmapResponse)
# # @log_exceptions
# # async def generate_mindmap(
# #     request: MindmapRequest,
# #     settings: Settings = Depends(get_settings)
# # ) -> MindmapResponse:
#     """Generate mindmap from PDF with improved error handling"""
#     logger.info(f"Generating mindmap for chat session: {request.chat_session_id}, source: {request.source_id}")
    
#     try:
#         user_id = validate_session(request.session_id, active_sessions)
#         chat_session_data = get_chat_session_by_id(request.chat_session_id, user_id)
        
#         if not chat_session_data:
#             raise ThreadNotFoundError()

#         source_data = next(
#             (s for s in chat_session_data.get("sources", []) 
#              if s.get("source_id") == request.source_id),
#             None
#         )
        
#         if not source_data or not source_data.get("filepath"):
#             raise SourceNotFoundError()

#         pdf_path = source_data["filepath"]
#         estimated_time = estimate_mindmap_generation_time(pdf_path)
#         logger.info(f"Estimated generation time: {estimated_time} seconds")

#         markdown, error = await generate_mindmap_from_pdf(pdf_path)
#         if error:
#             logger.error(f"Error generating mindmap: {error}")
#             raise HTTPException(status_code=500, detail=error)

#         # Save mindmap
#         mindmap_path = Path(pdf_path).parent / "mindmap.md"
#         mindmap_path.write_text(markdown, encoding="utf-8")
        
#         # Update database
#         update_source_field(
#             request.chat_session_id,
#             request.source_id,
#             {"mindmap.path": str(mindmap_path)}
#         )

#         logger.info(f"Mindmap generated and saved to: {mindmap_path}")
#         return MindmapResponse(
#             status="success",
#             markdown=markdown,
#             estimated_time=estimated_time,
#             chat_session_id=request.chat_session_id,
#             source_id=request.source_id
#         )

#     except Exception as e:
#         logger.error(f"Error in mindmap generation: {str(e)}", exc_info=True)
#         raise

 
# # @router.post("/podcast", response_model=PodcastResponse)
# # @log_exceptions
# # async def generate_podcast(
# #     request: PodcastRequest,
# #     settings: Settings = Depends(get_settings)
# # ) -> PodcastResponse:
# #     """Generate podcast from mindmap with improved error handling"""
#     logger.info(f"Generating podcast for chat session: {request.chat_session_id}, source: {request.source_id}")
    
#     try:
#         user_id = validate_session(request.session_id, active_sessions)
#         chat_session_data = get_chat_session_by_id(request.chat_session_id, user_id)
        
#         if not chat_session_data:
#             raise ThreadNotFoundError()

#         source_data = next(
#             (s for s in chat_session_data.get("sources", []) 
#              if s.get("source_id") == request.source_id),
#             None
#         )
        
#         if not source_data:
#             raise SourceNotFoundError()

#         # Try to read mindmap from disk if path exists
#         mindmap_md = None
#         mindmap_path = source_data.get("mindmap", {}).get("path")
#         if mindmap_path and Path(mindmap_path).exists():
#             mindmap_md = Path(mindmap_path).read_text(encoding="utf-8")
#         else:
#             logger.info("Mindmap not found, generating from PDF")
#             pdf_path = source_data.get("filepath")
#             if not pdf_path:
#                 raise SourceNotFoundError("PDF not found for source")
#             mindmap_md, error = await generate_mindmap_from_pdf(pdf_path)
#             if error:
#                 logger.error(f"Error generating mindmap: {error}")
#                 raise HTTPException(status_code=500, detail=error)

#         estimated_time = estimate_podcast_generation_time(mindmap_md)
#         logger.info(f"Estimated podcast generation time: {estimated_time} seconds")

#         logger.info("Starting podcast generation...")
#         podcast_data, error = await generate_podcast_from_mindmap(
#             mindmap_md
#         )
#         logger.info(f"Podcast generation complete. Error: {error}")
        
#         if error:
#             logger.error(f"Error generating podcast: {error}")
#             raise HTTPException(status_code=500, detail=error)

#         # Store audio data along with metadata
#         # podcast_data contains: script, audio_base64, audio_player
#         podcast_metadata = {
#             "status": "generated",
#             "generated_at": datetime.utcnow().isoformat(),
#             "script": podcast_data.get("script", ""),
#             "audio_base64": podcast_data.get("audio_base64", ""),
#             "has_audio": bool(podcast_data.get("audio_base64"))
#         }
        
#         logger.info(f"Updating database with podcast metadata for source {request.source_id}")
#         # Update database with podcast data (now includes audio)
#         update_source_field(
#             request.chat_session_id,
#             request.source_id,
#             {"podcast": podcast_metadata}
#         )
#         logger.info(f"Database updated successfully")

#         logger.info("Podcast generated successfully - preparing response")
#         response = PodcastResponse(
#             status="success",
#             data={
#                 "status": "generated",
#                 "script": podcast_data.get("script", ""),
#                 "audio_base64": podcast_data.get("audio_base64", ""),
#                 "generated_at": datetime.utcnow().isoformat()
#             },
#             estimated_time=estimated_time,
#             chat_session_id=request.chat_session_id,
#             source_id=request.source_id
#         )
#         logger.info(f"Returning podcast response with audio data")
#         return response

#     except Exception as e:
#         logger.error(f"Error in podcast generation: {str(e)}", exc_info=True)
#         raise