from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import routers
from services.pdf_service import router as pdf_router
from services.chat_service import router as chat_router
from services.mindmap_service import router as mindmap_router
#from services.websocket_handler import router as ws_router

# Import auth and config
from auth import router as auth_router
from config import get_settings
from logging_config import init_from_settings, get_logger

# Initialize logging before anything else
init_from_settings()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PDF Reader",
    description="Local PDF RAG Bot with Chat, Mindmap, and Podcast features",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(pdf_router, prefix="/pdf", tags=["PDF Operations"])
app.include_router(chat_router, prefix="/chat", tags=["Chat Operations"])
app.include_router(mindmap_router, prefix="/mindmap", tags=["Mindmap Operations"])
""" Disabled non-core features: podcast, websocket
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
"""

# Startup event
@app.on_event("startup")
async def startup_event():
    settings = get_settings()
    logger.info("Service started with modular architecture")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to  PDF Reader API"}

# Add server startup code
if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Reduce websocket warning noise
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    
    uvicorn.run(
        "pdfreader:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled reload to prevent constant reloading
        timeout_keep_alive=600,  # 10 minutes keep-alive for long operations
        log_level="info"
    )