# PDF RAG Chatbot Frontend (Streamlit)

Single-page Python app for uploading PDFs, chatting, generating mindmaps, and creating podcasts.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Backend running at `http://localhost:8000`

### Setup (Windows)

```bash
# 1. Navigate to frontend folder
cd frontend

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

### Configuration

Set environment variables in a `.env` file in the `frontend/` directory:

```env
API_URL=http://localhost:8000
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
```

Or export via terminal:

```bash
# Windows PowerShell
$env:API_URL="http://localhost:8000"

# Windows Command Prompt
set API_URL=http://localhost:8000
```

## 📖 Features

- **🔐 Authentication**: Register and login with email/password
- **📄 PDF Upload**: Upload multiple PDFs and chat with them
- **💬 Chat**: Real-time chat with context from uploaded documents
- **🧠 Mindmap Generation**: Auto-generate mindmaps from PDF content
- **🎙️ Podcast Generation**: Convert mindmaps to podcast scripts with audio
- **📁 Session Management**: Create new chat sessions, track uploads

## 🎨 UI Components

### Left Sidebar
- New chat session button
- PDF upload interface
- List of uploaded files
- Logout button

### Main Chat Area
- Message history (user/bot)
- Text input for messages
- Send message button
- Generate mindmap button
- Generate podcast button

## 🔌 API Integration

The app connects to the backend at:
- `POST /auth/login` — User login
- `POST /auth/register` — User registration
- `POST /pdf/upload` — Upload PDF
- `POST /chat/send` — Send chat message
- `POST /mindmap/generate` — Generate mindmap
- `POST /mindmap/podcast` — Generate podcast

All requests include the JWT token in the `Authorization` header.

## 📦 Dependencies

- **streamlit** — Web UI framework
- **requests** — HTTP client for API calls
- **python-dotenv** — Environment variable management

## 🐛 Troubleshooting

### Backend not reachable
```
ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```
- Check backend is running: `python pdfreader.py`
- Check API_URL matches: `API_BASE = "http://localhost:8000"`

### Port already in use
```
streamlit run app.py --server.port=8502
```

### File upload fails
- Check backend `/pdf/upload` endpoint is working
- Verify file size < max upload limit

## 📝 Example Workflow

1. **Register**: Click "Register", enter username/email/password
2. **Login**: Use credentials to login
3. **Upload PDF**: Use sidebar to upload a PDF file
4. **Chat**: Ask questions about the PDF in the chat box
5. **Mindmap**: Click "Generate Mindmap" to create a mindmap
6. **Podcast**: Click "Generate Podcast" to create audio

## 🎯 Development

To modify the UI, edit `app.py`. Streamlit reloads automatically on save.

Add new features:
1. Create new function (e.g., `def new_feature():`)
2. Call it from `chat_page()` or add a new button
3. Use `api_request()` helper to call backend endpoints

Example:
```python
if st.button("🆕 New Feature"):
    result, err = api_request("POST", "/api/new-feature", {"data": "value"})
    if result:
        st.success("Done!")
    else:
        st.error(f"Failed: {err}")
```

## 📞 Support

For API issues, check backend logs at `./logs/app.log`

For frontend issues, run with verbose logging:
```bash
streamlit run app.py --logger.level=debug
```

---

**Version**: 1.0.0  
**Last Updated**: October 18, 2024
