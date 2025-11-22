# Reflex UI for PDF RAG Chatbot

## Features
- User authentication (login)
- PDF upload and selection
- RAG chat interface (ask questions about uploaded PDFs)

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Reflex app:
   ```bash
   reflex init
   reflex run
   ```
3. Make sure your FastAPI backend is running at `http://localhost:8000`.

## Notes
- Update `BACKEND_URL` in `app.py` if your backend runs on a different address.
- This UI is minimal and can be extended for registration, session management, and better error handling.
- The app will be available at `http://localhost:3000` by default.
