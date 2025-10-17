import os
import re
import tempfile
import base64

import uvicorn
import numpy as np
import faiss
import fitz            # PyMuPDF
import pdfplumber
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

# === FastAPI Setup ===
app = FastAPI(title="LLAMA PDF Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ensure image directory exists
os.makedirs("static/images", exist_ok=True)

#=== Globals ===
pdf_chunks = []
faiss_index = None
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = OllamaLLM(model="llama3", temperature=0.1)

#=== Models ===
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

#=== Helpers ===
def chunk_text(text: str, max_len: int = 500) -> list[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + " "
        else:
            chunks.append(current.strip())
            current = s + " "
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 30]

def table_to_markdown(headers: list[str], rows: list[list]) -> str:
    header_line = " | ".join(headers)
    sep_line = " | ".join("―" * len(headers))
    row_lines = [" | ".join(str(cell or "") for cell in row) for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])

def extract_content(file_path: str, filename: str):
    text_chunks, table_chunks, image_chunks = [], [], []
    
    # 1) Extract text & tables via pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # text
            txt = page.extract_text() or ""
            for c in chunk_text(txt):
                text_chunks.append({
                    "text": c,
                    "filename": filename,
                    "type": "text",
                })
            # tables
            for table in page.extract_tables():
                if len(table) < 2:
                    continue
                headers, *rows = table
                headers = [h or f"Col{i}" for i, h in enumerate(headers)]
                md = table_to_markdown(headers, rows)
                for c in chunk_text(md):
                    table_chunks.append({
                        "text": c,
                        "filename": filename,
                        "type": "table",
                    })
    
    # 2) Extract images via PyMuPDF (in-memory, no saving)
    try:
        doc = fitz.open(file_path)
    except AttributeError:
        doc = fitz.Document(file_path)
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # convert CMYK→RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            img_bytes = pix.tobytes("png")
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            pix = None
            desc = f"[Figure: {filename} page {page_index+1}]"
            for c in chunk_text(desc, max_len=200):
                image_chunks.append({
                    "text":    c,
                    "filename": filename,
                    "type":     "image",
                    "image_b64": img_b64,  # in-memory base64
                })

    # Combine
    all_chunks = text_chunks + table_chunks + image_chunks
    return all_chunks

def build_faiss_index(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index

def find_relevant_chunks(question: str, chunks: list[dict], index, top_k: int = 3):
    q_emb = embedding_model.encode([question]).astype('float32')
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

def ask_llama(question: str, chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        if c["type"] == "image":
            # Just mention image presence, optionally include base64 or a placeholder
            parts.append(f"Image extracted from [{c['filename']}], base64 available.")
        else:
            parts.append(f"Source [{c['filename']}]:\n{c['text']}")
    context = "\n\n".join(parts)
    prompt = f"""
You are a helpful assistant. Use the following context to answer the question accurately.
{context}

Question: {question}
Answer:
"""
    resp = llm.generate([prompt])
    return resp.generations[0][0].text.strip()

#=== Endpoints ===
@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global pdf_chunks, faiss_index
    all_chunks = []
    
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files supported.")
        # save temp
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        # extract
        chunks = extract_content(tmp_path, file.filename)
        all_chunks.extend(chunks)
        os.remove(tmp_path)
    
    pdf_chunks = all_chunks
    faiss_index = build_faiss_index(pdf_chunks)
    return {
        "message": f"Processed {len(files)} file(s)",
        "chunks": len(pdf_chunks),
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    if not pdf_chunks or faiss_index is None:
        raise HTTPException(400, "No data. Please upload PDFs first.")
    
    top_chunks = find_relevant_chunks(req.question, pdf_chunks, faiss_index, top_k=req.top_k)
    answer = ask_llama(req.question, top_chunks)
    sources = list({c["filename"] for c in top_chunks})
    return AnswerResponse(answer=answer, sources=sources)

@app.get("/")
async def serve_ui():
    return FileResponse(os.path.join("static", "index.html"))

#=== Run Server ===
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)