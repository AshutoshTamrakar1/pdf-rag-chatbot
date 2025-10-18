import logging
import asyncio
import os
import fitz         # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, AsyncGenerator, Optional, Tuple

from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

# --- RAG and Model Setup ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pdf_rag_cache = {}  # Cache for processed PDFs to avoid reprocessing

# --- Ollama Local Models ---
llama3_llm = OllamaLLM(model="llama3", temperature=0.1)
gemma_llm = OllamaLLM(model="gemma", temperature=0.1)
phi3_llm = OllamaLLM(model="phi3", temperature=0.1)

# --- Podcast Generation ---
PODCAST_SCRIPT_PROMPT_TEMPLATE = """You are a professional podcast scriptwriter. Create an engaging, informative dialogue between two podcast hosts named Jess and Leo discussing the topics below.

RULES:
1. Write natural, conversational dialogue that explains concepts clearly
2. Include questions and follow-up discussions that build understanding
3. Use proper grammar and punctuation
4. Format EXACTLY as: Jess: [dialogue] Leo: [dialogue]
5. Each speaker should have 2-4 sentences per turn
6. Make it educational yet entertaining - like a real podcast
7. Include brief [laughs] or [pause] for natural flow
8. Start with a greeting: "Welcome to our podcast about..."
9. End with: "Thanks for listening! If you'd like to implement this content in your application, contact our team."

TOPICS TO DISCUSS:
{mindmap_md}

IMPORTANT: Create a coherent discussion that covers the main points from the topics above. Make it sound like real podcast hosts discussing this topic naturally.

Now write the podcast script:
"""

# --- Estimation functions ---
def estimate_mindmap_generation_time(pdf_path: str) -> int:
    try:
        pdf_text = _extract_text_from_pdf(pdf_path)
        char_count = len(pdf_text)
        base_time_seconds = 15
        chars_per_second_of_processing = 1000
        estimated_time = base_time_seconds + (char_count / chars_per_second_of_processing)
        max_time_seconds = 300
        final_time = min(int(estimated_time), max_time_seconds)
        logger.info(f"Estimated mindmap generation time for PDF with {char_count} chars: {final_time}s")
        return final_time
    except Exception as e:
        logger.error(f"Could not estimate mindmap time for {pdf_path}: {e}")
        return 45

def estimate_podcast_generation_time(mindmap_md: str) -> int:
    if not mindmap_md:
        return 45
    char_count = len(mindmap_md)
    line_count = mindmap_md.count('\n') + 1
    base_time_seconds = 30
    time_per_char = 1/12
    time_per_line = 0.5
    estimated_time = base_time_seconds + (char_count * time_per_char) + (line_count * time_per_line)
    max_time_seconds = 480
    final_time = min(int(estimated_time), max_time_seconds)
    logger.info(
        f"Estimated podcast generation time for mindmap with {char_count} chars and "
        f"{line_count} lines: {final_time}s"
    )
    return final_time

# --- RAG Helper Functions ---
def _extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def _chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ","",""])
    return text_splitter.split_text(text)

def _get_top_k_chunks(query: str, chunks: List[str], embeddings: torch.Tensor, k: int = 3) -> str:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(chunks)))
    return "\n\n--\n\n".join(chunks[i] for i in top_results.indices)

# --- Chat Functions ---
async def chat_completion_LlamaModel_ws(text: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Standard Llama3 WS completion for text: '{text[:50]}...'")
    try:
        messages_to_send = history[-(HISTORY_LENGTH * 2):]
        if messages_to_send and messages_to_send[0]['role'] == 'assistant':
            messages_to_send.pop(0)
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_send]) + f"\nuser: {text}"
        response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Llama3 Chat: {e}", exc_info=True)
        yield None, f"Error during Llama3 chat completion: {e}"

async def chat_completion_Gemma_ws(text: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Gemma WS completion for text: '{text[:50]}...'")
    try:
        messages_to_send = history[-(HISTORY_LENGTH * 2):]
        if messages_to_send and messages_to_send[0]['role'] == 'assistant':
            messages_to_send.pop(0)
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_send]) + f"\nuser: {text}"
        response = await asyncio.to_thread(gemma_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Gemma Chat: {e}", exc_info=True)
        yield None, f"Error during Gemma chat completion: {e}"

async def chat_completion_with_pdf_ws(text: str, history: List[Dict[str, str]], pdf_path: str) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Main Chat RAG completion for text: {text[:50]}... using single PDF: {pdf_path}")
    try:
        if pdf_path not in pdf_rag_cache:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF not found at path for Main Chat RAG: {pdf_path}")
                yield None, f"The associated document could not be found."
                return
            pdf_text = _extract_text_from_pdf(pdf_path)
            chunks = _chunk_text(pdf_text)
            embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
            pdf_rag_cache[pdf_path] = {"chunks": chunks, "embeddings": embeddings}
        cached_data = pdf_rag_cache[pdf_path]
        context = _get_top_k_chunks(text, cached_data["chunks"], cached_data["embeddings"])
        rag_history = history[-(HISTORY_LENGTH * 2):]
        if rag_history and rag_history[0]['role'] == 'assistant':
            rag_history.pop(0)
        prompt = f"Context:\n{context}\n\nQuestion: {text}\nAnswer:"
        response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Main Chat RAG: {e}", exc_info=True)
        yield None, f"Error during Main Chat RAG completion: {e}"

async def chat_completion_with_multiple_pdfs_ws(text: str, history: List[Dict[str, str]], pdf_paths: List[str]) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Multi-PDF RAG for text: {text[:50]}... on {len(pdf_paths)} documents.")
    try:
        all_context_chunks = []
        for pdf_path in pdf_paths:
            if not pdf_path or not os.path.exists(pdf_path):
                logger.warning(f"skipping non-existent PDF for multi-RAG: {pdf_path}")
                continue
            if pdf_path not in pdf_rag_cache:
                pdf_text = _extract_text_from_pdf(pdf_path)
                chunks = _chunk_text(pdf_text)
                embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
                pdf_rag_cache[pdf_path] = {"chunks": chunks, "embeddings": embeddings}
            cached_data = pdf_rag_cache[pdf_path]
            context = _get_top_k_chunks(text, cached_data["chunks"], cached_data["embeddings"], k=2)
            all_context_chunks.append(f"Context from {os.path.basename(pdf_path)}:\n{context}")
        if not all_context_chunks:
            yield None, "No valid documents could be processed for context."
            return
        combined_context = "\n\n--\n\n".join(all_context_chunks)
        rag_history = history[-(HISTORY_LENGTH * 2):]
        if rag_history and rag_history[0]['role'] == 'assistant':
            rag_history.pop(0)
        prompt = f"Context from multiple documents:\n{combined_context}\n\nQuestion: {text}\nAnswer:"
        response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Multi-PDF RAG: {e}", exc_info=True)
        yield None, f"Error during Multi-PDF RAG completion: {e}"

# --- Mindmap Generation using phi3 ---
async def generate_mindmap_from_pdf(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
    logger.info(f"MindmapGen: Processing PDF at path: {pdf_path}")
    if not os.path.exists(pdf_path):
        err = f"PDF file not found at path: {pdf_path}"
        logger.error(f"MindmapGen: {err}")
        return None, err
    try:
        pdf_text = _extract_text_from_pdf(pdf_path)
        prompt = f"Generate a mindmap markdown for the following document:\n{pdf_text}"
        response = await asyncio.to_thread(phi3_llm.generate, [prompt])
        mindmap_md = response.generations[0][0].text.strip()
        logger.info("MindmapGen: Successfully generated mindmap markdown.")
        return mindmap_md, None
    except Exception as e:
        err = f"An unexpected error occurred during mindmap generation: {e}"
        logger.error(f"MindmapGen: {err}", exc_info=True)
        return None, "An internal error occurred while generating the mindmap."

# --- Podcast Generation using phi3 + local TTS ---
import base64
import tempfile
from services.local_audio import generate_tts_audio

async def generate_podcast_from_mindmap(mindmap_md: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    logger.info("PodcastGen: Generating podcast script from mindmap markdown.")
    script_prompt = PODCAST_SCRIPT_PROMPT_TEMPLATE.format(mindmap_md=mindmap_md)
    try:
        # Use llama3 instead of phi3 for better coherence in long-form content
        response = await asyncio.to_thread(llama3_llm.generate, [script_prompt])
        script = response.generations[0][0].text.strip()
        logger.info("PodcastGen: Script generated successfully.")
    except Exception as e:
        err = f"Failed to generate podcast script: {e}"
        logger.error(f"PodcastGen: {err}", exc_info=True)
        return None, err

    logger.info("PodcastGen: Generating podcast audio locally.")
    tmp_path = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp_file.name
        tmp_file.close()

        # Synthesize audio using blocking TTS on a thread
        await asyncio.to_thread(generate_tts_audio, script, tmp_path)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        audio_player_html = f'<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>'

        logger.info("PodcastGen: Local audio generated successfully.")
        return {"script": script, "audio_base64": audio_base64, "audio_player": audio_player_html}, None
    except Exception as e:
        err = f"An unexpected error occurred during local audio generation: {e}"
        logger.error(f"PodcastGen: {err}", exc_info=True)
        return None, "An internal error occurred while generating the podcast audio."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --- Title Generation using llama3 ---
async def generate_chat_title(messages_for_title_generation: List[Dict[str, str]]) -> Optional[str]:
    if not messages_for_title_generation:
        return None
    logger.info(f"Generating chat title for a conversation with {len(messages_for_title_generation)} messages.")
    conversation_summary = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages_for_title_generation)
    prompt = TITLE_GENERATION_PROMPT.format(conversation_text=conversation_summary)
    try:
        response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        title = response.generations[0][0].text.strip().strip('\'"')
        if title:
            logger.info(f"Successfully generated chat title: '{title}'")
            return title
        else:
            logger.warning("Title generation resulted in an empty string.")
            return None
    except Exception as e:
        logger.error(f"Failed to generate chat title: {e}", exc_info=True)
        return None

# --- Constants ---
SYSTEM_PROMPT  = """You are a helpful Al assistant. Provide concise and accurate answers based on the conversation history."""

RAG_SYSTEM_PROMPT = """ You are an expert assistant. Use ONLY the provided context to answer the user's question accurately. 
If the answer is not in the context, say "I cannot answer this based on the provided document." Do not use any prior knowledge."""

TITLE_GENERATION_PROMPT = """Based on the following conversation, generate a short, concise title (4-5 words max).
Do not use any quotation marks or labels in your response. Just provide the title text.

CONVERSATION:
{conversation_text}
"""

HISTORY_LENGTH = 10

logger.info("ai_engine.py loaded with local Ollama models: llama3, gemma, phi3.")