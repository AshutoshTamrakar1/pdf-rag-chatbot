# Local PDF RAG Bot

A FastAPI-powered chatbot that lets you upload PDF files, ask questions, and get answers using local retrieval-augmented generation (RAG) with Llama 3 and sentence-transformers. No cloud storage or external file saving—everything is processed in memory.

## Features

- **PDF Upload:** Drag and drop multiple PDFs for instant processing.
- **Text, Table, and Image Extraction:** Extracts text, tables (as markdown), and images (base64, in-memory) from PDFs.
- **Semantic Search:** Uses FAISS and sentence-transformers for fast, relevant chunk retrieval.
- **LLM Answers:** Answers questions using context from your PDFs via Llama 3 (Ollama).
- **Sources:** Shows which PDF(s) the answer was based on.
- **No Local Saving:** Images and data are kept in memory, not saved to disk.

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama (for Llama 3)

Make sure [Ollama](https://ollama.com/) is running and the `llama3` model is available.

### 3. Run the server

```bash
python app.py
```

The server will start at [http://127.0.0.1:8001](http://127.0.0.1:8001).

### 4. Open the UI

Open `static/index.html` in your browser, or visit [http://127.0.0.1:8001](http://127.0.0.1:8001) if running locally.

## Usage

1. **Upload PDFs:** Click "Upload", select your PDF files, and wait for processing.
2. **Ask Questions:** Type your question and choose how many top chunks to use (Top K).
3. **View Answers:** The bot responds with answers and source files.

## Project Structure

```
local-pdf-rag-bot/
├── app.py                # FastAPI backend
├── requirements.txt      # Python dependencies
├── static/
│   └── index.html        # Frontend UI
```

## Dependencies

- FastAPI, Uvicorn
- pdfplumber, PyMuPDF, Pillow
- sentence-transformers
- faiss-cpu
- langchain, langchain-ollama
- pandas (for table handling)
- Ollama (external, for Llama 3)

See `requirements.txt` for details.

## Notes

- No PDF/image data is saved to disk (except temporary files for extraction).
- Images are handled as base64 strings in memory.
- For best results, use clear, well-structured PDFs.
