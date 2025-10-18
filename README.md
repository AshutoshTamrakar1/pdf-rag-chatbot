# PDF RAG Chatbot v2.0

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that processes PDF documents locally, enabling intelligent conversations with your documents. Built with FastAPI, MongoDB, and Ollama LLMs, featuring real-time chat, mindmap generation, and podcast creation capabilities.

![Status](https://img.shields.io/badge/status-active-brightgreen)
![Version](https://img.shields.io/badge/version-2.0.0-blue)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [WebSocket API](#websocket-api)
- [Project Structure](#project-structure)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### Core Functionality
- **🤖 RAG-Powered Chat**: Ask questions about uploaded PDF documents with context-aware responses
- **📊 Mindmap Generation**: Automatically generate mindmaps from PDF content
- **🎙️ Podcast Creation**: Convert document summaries into podcast scripts with audio generation
- **🔐 User Authentication**: Secure JWT-based authentication with session management
- **💾 Chat History**: Persistent storage of conversations and sessions
- **📁 Multi-PDF Support**: Chat with context from multiple documents simultaneously

### Technical Features
- **Real-time Communication**: WebSocket support for live updates
- **Local LLM Integration**: Uses Ollama for privacy-preserving AI processing
- **Semantic Search**: Sentence Transformers for intelligent document chunking and retrieval
- **Cross-Platform Audio**: TTS/STT capabilities with local models
- **Async Processing**: Non-blocking operations for better performance
- **Comprehensive Logging**: Centralized logging with error tracking

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Client                         │
│                  (Web/Desktop App)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼ HTTP       ▼ WebSocket  ▼
    REST API    Streaming Data   Commands
        │            │            │
┌───────────────────────────────────────────────────────┐
│                   FastAPI Server                      │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐   │
│  │ Auth Router  │  │ PDF Router  │  │Chat Router │   │
│  └──────────────┘  └─────────────┘  └────────────┘   │
│  ┌──────────────┐  ┌─────────────┐                   │
│  │  WS Handler  │  │WebSocket API│  ┌────────────┐   │
│  └──────────────┘  └─────────────┘  │Mindmap API │   │
│                                      └────────────┘   │
└───────────┬───────────────────────────────────────────┘
            │
   ┌────────┴─────────┐
   │                  │
   ▼                  ▼
┌──────────────┐  ┌──────────────┐
│  MongoDB     │  │ Local Models │
│  (Sessions,  │  │              │
│   Chat Data) │  │ • Ollama LLM │
└──────────────┘  │ • Embeddings │
                  │ • TTS/STT    │
                  └──────────────┘
```

### Data Flow

```
User Upload PDF
       ↓
   Extract Text
       ↓
   Chunk Content (1000 char chunks, 200 char overlap)
       ↓
   Generate Embeddings (SentenceTransformer)
       ↓
   Store in Cache (pdf_rag_cache)
       ↓
User Asks Question
       ↓
   Generate Query Embedding
       ↓
   Semantic Search (top 3 chunks)
       ↓
   Create Context Window
       ↓
   Send to Ollama LLM
       ↓
   Stream/Return Response
```

### Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Authentication Layer** | User registration, login, JWT tokens, session management | FastAPI, JWT, bcrypt |
| **Database Layer** | Persistent storage of users, sessions, documents, messages | MongoDB |
| **RAG Engine** | Document processing, chunking, semantic search, context retrieval | PyMuPDF, SentenceTransformer, LangChain |
| **LLM Interface** | Integration with local language models | Ollama, LangChain-Ollama |
| **WebSocket Handler** | Real-time bidirectional communication | FastAPI WebSockets |
| **Audio Service** | Text-to-speech and speech-to-text | pyttsx3, faster-whisper |
| **Configuration Manager** | Environment-based settings | Pydantic, python-dotenv |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- MongoDB (local or Atlas)
- Ollama (for local LLMs)
- 8GB+ RAM (for embedding models)
- CUDA-capable GPU (optional, for faster processing)

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your configuration

# 5. Install and start Ollama
# Download from https://ollama.ai
ollama pull llama3
ollama serve  # In a separate terminal

# 6. Start MongoDB (if local)
mongod

# 7. Run the application
python -m uvicorn pdfreader:app --reload

# 8. Access the API
# API Docs: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

## 📦 Installation

### Full Setup with All Features

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# For audio processing (system dependencies)
# Linux
sudo apt-get install portaudio19-dev ffmpeg

# macOS
brew install portaudio ffmpeg

# Windows
# Download ffmpeg from https://ffmpeg.org and add to PATH
```

### Install Ollama and Models

```bash
# 1. Download and install Ollama from https://ollama.ai

# 2. Pull required models
ollama pull llama3       # Main model (~4GB)
ollama pull gemma        # Secondary model (~3GB)
ollama pull phi3         # Lightweight model (~1GB)

# 3. Start Ollama service
ollama serve             # Listens on http://localhost:11434
```

### Setup MongoDB

```bash
# Option 1: Local MongoDB
# Download and install from https://www.mongodb.com/try/download/community
mongod                   # Start local instance

# Option 2: MongoDB Atlas (Cloud)
# 1. Create account at https://www.mongodb.com/cloud/atlas
# 2. Create a cluster
# 3. Get connection string
# 4. Set in .env (see .env.example for format)
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# Application
APP_NAME=PDF RAG Chatbot
ENVIRONMENT=development    # development, staging, production
DEBUG=true

# Server
HOST=0.0.0.0
PORT=8000

# Database
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=pdf_rag_chatbot

# Authentication (Change in production!)
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480
REFRESH_TOKEN_EXPIRE_DAYS=7

# AI Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MAIN_MODEL=llama3
EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Parameters
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=3

# Audio
TTS_RATE=150
TTS_VOLUME=0.9
STT_MODEL=base

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Generate Secure JWT Secret

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Output: copy-this-to-JWT_SECRET_KEY
```

## 📚 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123",
  "username": "johndoe"
}

Response:
{
  "user_id": "uuid",
  "email": "user@example.com",
  "username": "johndoe",
  "created_at": "2024-10-18T10:30:00"
}
```

#### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword123"
}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user_id": "uuid",
  "expires_in": 28800
}
```

#### Refresh Token
```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}

Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "expires_in": 28800
}
```

### PDF Operations

#### Upload PDF
```http
POST /pdf/upload
Authorization: Bearer <access_token>
Content-Type: multipart/form-data

{
  "file": <pdf-file>,
  "chat_session_id": "uuid",
  "filename": "document.pdf"
}

Response:
{
  "source_id": "uuid",
  "filename": "document.pdf",
  "size_bytes": 1024000,
  "pages": 25,
  "upload_status": "success"
}
```

### Chat Endpoints

#### Send Chat Message
```http
POST /chat/send
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "chat_session_id": "uuid",
  "question": "What is the main topic of this document?",
  "source_ids": ["uuid1", "uuid2"]
}

Response:
{
  "response": "The main topic of this document is...",
  "sources_cited": [
    {
      "source_id": "uuid1",
      "page": 5,
      "excerpt": "..."
    }
  ],
  "model_used": "llama3",
  "processing_time_ms": 1240
}
```

#### Get Chat History
```http
GET /chat/sessions/{user_id}
Authorization: Bearer <access_token>

Response:
{
  "chat_sessions": [
    {
      "session_id": "uuid",
      "title": "Document Analysis",
      "created_at": "2024-10-18T10:30:00",
      "message_count": 15,
      "sources_count": 2
    }
  ]
}
```

### Mindmap & Podcast Endpoints

#### Generate Mindmap
```http
POST /mindmap/generate
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "chat_session_id": "uuid",
  "source_ids": ["uuid1", "uuid2"]
}

Response:
{
  "mindmap_id": "uuid",
  "mindmap_md": "# Title\n## Section 1\n- Point 1\n- Point 2",
  "generation_time_ms": 5000,
  "status": "success"
}
```

#### Generate Podcast
```http
POST /mindmap/podcast
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "mindmap_md": "# Title\n## Topic\n- Point 1",
  "chat_session_id": "uuid"
}

Response:
{
  "podcast_id": "uuid",
  "audio_url": "/podcasts/podcast_uuid.mp3",
  "duration_seconds": 120,
  "status": "success"
}
```

## 🔌 WebSocket API

### Connection

```javascript
// Connect
const ws = new WebSocket(
  `ws://localhost:8000/ws/ws?session_id=${sessionId}`
);

ws.addEventListener('open', (event) => {
  console.log('Connected to WebSocket');
});
```

### Message Types

#### Create New Thread
```json
{
  "type": "create_new_thread"
}

// Response
{
  "type": "thread_created",
  "chat_session_id": "uuid",
  "title": "New Chat"
}
```

#### Send Chat Message
```json
{
  "type": "chat_message",
  "payload": {
    "chat_session_id": "uuid",
    "question": "Explain this concept",
    "source_ids": ["uuid1"]
  }
}

// Response
{
  "type": "chat_response",
  "response": "The concept of...",
  "chat_session_id": "uuid",
  "timestamp": "2024-10-18T10:30:00"
}
```

#### Generate Mindmap
```json
{
  "type": "generate_mindmap",
  "payload": {
    "chat_session_id": "uuid",
    "source_ids": ["uuid1"]
  }
}

// Response
{
  "type": "mindmap_response",
  "mindmap": "# Topic\n## Section\n- Point",
  "chat_session_id": "uuid",
  "timestamp": "2024-10-18T10:30:00"
}
```

#### Generate Podcast
```json
{
  "type": "generate_podcast",
  "payload": {
    "mindmap_md": "# Content",
    "chat_session_id": "uuid"
  }
}

// Response
{
  "type": "podcast_response",
  "audio_url": "/podcasts/audio.mp3",
  "duration": 120,
  "title": "Generated Podcast",
  "timestamp": "2024-10-18T10:30:00"
}
```

## 📁 Project Structure

```
pdf-rag-chatbot-v2/
├── pdfreader.py                 # Main FastAPI application
├── auth.py                      # Authentication logic & JWT management
├── config.py                    # Configuration management
├── db_manager.py               # MongoDB abstraction layer
├── ai_engine.py                # RAG & LLM integration
├── logging_config.py           # Centralized logging setup
│
├── services/
│   ├── chat_service.py         # Chat message handling
│   ├── pdf_service.py          # PDF upload & processing
│   ├── mindmap_service.py      # Mindmap generation
│   ├── websocket_handler.py    # WebSocket connection management
│   ├── stt_service.py          # Speech-to-text service
│   ├── podcast_service.py      # Podcast endpoint
│   ├── local_audio.py          # Audio TTS/STT functions
│   ├── models.py               # Pydantic request/response models
│   ├── exceptions.py           # Custom exception classes
│   ├── utils.py                # Utility functions
│   └── logging.py              # Service-level logging setup
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
└── uploads/                   # User-uploaded PDF files
    └── podcasts/              # Generated podcast audio files
```

### Key Files Overview

| File | Purpose | Key Functions |
|------|---------|--------------|
| `pdfreader.py` | Main app entry point | FastAPI app initialization, route registration |
| `auth.py` | Authentication system | User registration/login, JWT tokens, session management |
| `db_manager.py` | Database abstraction | MongoDB CRUD operations, data persistence |
| `ai_engine.py` | AI/ML operations | PDF processing, embedding, LLM integration, RAG |
| `config.py` | Configuration | Environment loading, settings validation |
| `services/chat_service.py` | Chat logic | Message handling, context management |
| `services/pdf_service.py` | PDF handling | Upload, parsing, storage |
| `services/websocket_handler.py` | WebSocket | Real-time connections, message routing |

## 🔧 Development

### Local Development Setup

```bash
# Install with development extras
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8 mypy

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
pytest tests/ -v

# Code formatting
black .

# Linting
flake8 . --max-line-length=100

# Type checking
mypy .
```

### Common Development Tasks

#### Adding a New API Endpoint

```python
# 1. Create request/response models in services/models.py
from pydantic import BaseModel

class MyRequest(BaseModel):
    param1: str
    param2: int

class MyResponse(BaseModel):
    result: str

# 2. Create router in services/my_service.py
from fastapi import APIRouter, Depends
from logging_config import get_logger, log_exceptions

router = APIRouter()
logger = get_logger(__name__)

@router.post("/endpoint", response_model=MyResponse)
@log_exceptions(logger)
async def my_endpoint(request: MyRequest):
    # Implementation
    return MyResponse(result="...")

# 3. Include in pdfreader.py
app.include_router(my_router, prefix="/api", tags=["My Feature"])
```

#### Adding a New Database Operation

```python
# In db_manager.py
from logging_config import log_exceptions

@log_exceptions(logger)
def my_query(param: str) -> dict:
    """Query description"""
    collection = MongoDBManager().db[COLLECTION_NAME]
    result = collection.find_one({"key": param})
    return result or {}
```

### Debugging

#### Enable Verbose Logging

```env
LOG_LEVEL=DEBUG
```

#### MongoDB Connection Issues

```python
# In Python REPL
from db_manager import MongoDBManager
manager = MongoDBManager()
manager.check_database_connection()
```

#### Ollama Model Issues

```bash
# Check running models
curl http://localhost:11434/api/tags

# Test LLM
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama3", "prompt": "Hello", "stream": false}'
```

## 🐛 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Error
```
Error: ServerSelectionTimeoutError
```

**Solution:**
```bash
# Check MongoDB is running
mongod --version
mongod --dbpath /path/to/data  # Start MongoDB

# Verify connection string in .env
# MONGODB_URI=mongodb://localhost:27017
```

#### 2. Ollama Not Found
```
Error: Failed to connect to Ollama at http://localhost:11434
```

**Solution:**
```bash
# Install Ollama from https://ollama.ai
# Start service
ollama serve

# Verify models are pulled
ollama list
ollama pull llama3  # If needed
```

#### 3. Out of Memory Error
```
CUDA out of memory or insufficient RAM
```

**Solution:**
- Reduce `RAG_CHUNK_SIZE` in .env
- Use smaller Ollama models (phi3 instead of llama3)
- Close other applications
- Increase system RAM/swap

#### 4. Slow Embedding Generation
**Solution:**
- If GPU available: CUDA should be auto-detected
- Check GPU availability: `torch.cuda.is_available()`
- Pre-compute and cache embeddings

#### 5. JWT Token Expired
```
Error: "Token has expired"
```

**Solution:**
```python
# Use refresh endpoint to get new token
POST /auth/refresh
{
  "refresh_token": "..."
}
```

### Debug Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment activated
- [ ] `pip install -r requirements.txt` successful
- [ ] MongoDB running and accessible
- [ ] Ollama running and models pulled
- [ ] `.env` configured correctly
- [ ] `LOG_LEVEL=DEBUG` for verbose output
- [ ] Check logs: `./logs/app.log`

## 📊 Performance Optimization

### Production Recommendations

1. **Database**
   - Use MongoDB Atlas for scalability
   - Enable read replicas for high availability
   - Regular backups and monitoring

2. **Caching**
   - Implement Redis for session storage (currently in-memory)
   - Cache frequently used embeddings
   - Use Celery for background tasks

3. **LLM**
   - Run Ollama on separate GPU server
   - Load-balance across multiple instances
   - Use smaller models for less critical tasks

4. **API**
   - Use Gunicorn/Uvicorn production server
   - Configure multiple workers
   - Enable gzip compression
   - Use CDN for static files

5. **Monitoring**
   - Implement health checks
   - Log to centralized service (ELK, DataDog)
   - Monitor GPU/CPU usage
   - Set up alerts for errors

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features
- Update README for significant changes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Ashutosh Tamrakar** - Initial development

## 🙏 Acknowledgments

- FastAPI community for the excellent framework
- Ollama for local LLM capabilities
- MongoDB for reliable data storage
- Sentence Transformers for embedding models
- All contributors and supporters

## 📞 Support

For issues, questions, or suggestions:

1. **GitHub Issues**: [Create an issue](https://github.com/yourusername/pdf-rag-chatbot/issues)
2. **Documentation**: Check the [Wiki](https://github.com/yourusername/pdf-rag-chatbot/wiki)
3. **Email**: your.email@example.com

---

**Last Updated**: October 18, 2024  
**Current Version**: 2.0.0  
**Status**: Active Development