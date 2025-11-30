import logging
import httpx
import asyncio
import os
import fitz         # PyMuPDF
import torch

#Compatibility shim: sentence-transformers imports `cached_download` from
#huggingface_hub in some versions. Some environments have a newer/older
#huggingface_hub that doesn't expose cached_download directly. Add a
#fallback shim so imports work regardless of installed huggingface_hub.
import importlib
_hf = None
try:
    _hf = importlib.import_module("huggingface_hub")
    # If cached_download is missing, provide a backward-compatible shim that
    # accepts the broad kwargs sentence-transformers may pass (including 'url').
    if not getattr(_hf, "cached_download", None):
        from huggingface_hub import hf_hub_download
        import inspect
        import requests

        def _cached_download_shim(*args, **kwargs):
            """Compatibility shim for huggingface_hub.cached_download.

            - If caller provides `url=...` we download the URL directly using requests
              (this is what older cached_download implementations supported).
            - Otherwise we delegate to hf_hub_download and filter kwargs to the
              signature supported by the installed huggingface_hub.
            """
            url = kwargs.get("url")
            if url:
                # Determine filename preference (sentence-transformers sometimes uses 'force_filename')
                filename = kwargs.get("force_filename") or kwargs.get("filename") or os.path.basename(url.split("?")[0])
                cache_dir = kwargs.get("cache_dir") or os.getcwd()
                os.makedirs(cache_dir, exist_ok=True)
                target_path = os.path.join(cache_dir, filename)

                force = kwargs.get("force_download") or kwargs.get("force", False)
                if os.path.exists(target_path) and not force:
                    return target_path

                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(target_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
                return target_path

            # Delegate to hf_hub_download — only pass args that the function supports
            sig = inspect.signature(hf_hub_download)
            acceptable = {k: v for k, v in kwargs.items() if k in sig.parameters}
            return hf_hub_download(*args, **acceptable)

        setattr(_hf, "cached_download", _cached_download_shim)
except Exception:
    # huggingface_hub may not be installed yet — let the normal import errors surface later
    _hf = None

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

# Available models mapping
AVAILABLE_MODELS = {
    "llama3": llama3_llm,
    "gemma": gemma_llm,
    "phi3": phi3_llm
}

# Chat models only (phi3 reserved for podcast generation)
CHAT_MODELS = {
    "llama3": llama3_llm,
    "gemma": gemma_llm
}

def get_available_models() -> List[str]:
    """Return list of available chat model names (excludes phi3 which is for podcasts)"""
    return list(CHAT_MODELS.keys())

def get_model_llm(model_name: str):
    """Get LLM instance by model name, defaults to llama3"""
    return CHAT_MODELS.get(model_name, llama3_llm)

def get_podcast_model():
    """Get the dedicated podcast generation model (phi3)"""
    return phi3_llm

# --- Podcast Generation (DISABLED) ---
'''
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
'''

# --- Mindmap Generation Prompt ---
MINDMAP_PROMPT_TEMPLATE = """You are an expert at creating hierarchical mindmaps. Analyze the following document text and create a comprehensive mindmap in Mermaid diagram syntax.

REQUIREMENTS:
1. Use ONLY Mermaid mindmap syntax - start with "mindmap" on first line
2. Use proper indentation with 2 spaces per level
3. Root node format: root((Title))
4. Create 3-5 main branches (key topics)
5. Each main branch should have 2-3 sub-branches
6. Keep labels concise (3-7 words max)
7. Output ONLY valid Mermaid code - no explanations, no markdown formatting

MERMAID SYNTAX (FOLLOW THIS EXACTLY):
mindmap
  root((Central Topic))
    Main Topic 1
      Subtopic 1.1
      Subtopic 1.2
    Main Topic 2
      Subtopic 2.1
      Subtopic 2.2

DOCUMENT TEXT:
{pdf_text}

Generate ONLY the Mermaid mindmap code:
"""

# --- Estimation functions ---
def estimate_mindmap_generation_time(pdf_path: str) -> int:
    try:
        # Import helper at runtime to avoid circular dependency
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        
        # Build prompt with system instructions and conversation history
        prompt = f"{SYSTEM_PROMPT}\n\n"
        prompt += "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_send])
        prompt += f"\nuser: {text}\nassistant:"
        
        try:
            response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        except Exception as e:
            # Detect connection problems to the local LLM service and return
            # a clearer error that instructs the operator how to fix it.
            if isinstance(e, (httpx.ConnectError, ConnectionRefusedError, OSError)):
                logger.error(f"LLM service connection error for Main Chat RAG: {e}")
                yield None, (
                    "LLM service unreachable. Ensure the local LLM service (e.g. Ollama) is running "
                    "and listening (default: localhost:11434). If you're running in Docker or remote host, update the client URL accordingly."
                )
                return
            # otherwise re-raise as an informative error
            logger.error(f"Main Chat RAG generation failed: {e}", exc_info=True)
            yield None, f"Error during Main Chat RAG completion: {e}"
            return
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
        
        # Build prompt with system instructions and conversation history
        prompt = f"{SYSTEM_PROMPT}\n\n"
        prompt += "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_send])
        prompt += f"\nuser: {text}\nassistant:"
        
        response = await asyncio.to_thread(gemma_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Gemma Chat: {e}", exc_info=True)
        yield None, f"Error during Gemma chat completion: {e}"

# --- Phi3 Completion (RESERVED FOR PODCAST GENERATION ONLY) ---
async def chat_completion_phi3_ws(text: str, history: List[Dict[str, str]]) -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Phi3 WS completion for text: '{text[:50]}...'")
    try:
        messages_to_send = history[-(HISTORY_LENGTH * 2):]
        if messages_to_send and messages_to_send[0]['role'] == 'assistant':
            messages_to_send.pop(0)
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_send]) + f"\nuser: {text}"
        response = await asyncio.to_thread(phi3_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Phi3 Chat: {e}", exc_info=True)
        yield None, f"Error during Phi3 chat completion: {e}"

async def chat_completion_with_pdf_ws(text: str, history: List[Dict[str, str]], pdf_path: str, model: str = "llama3") -> AsyncGenerator[Tuple[Optional[str], Optional[str]], None]:
    logger.info(f"Initiating Main Chat RAG completion for text: {text[:50]} using single PDF: {pdf_path} with model: {model}")
    try:
        if pdf_path not in pdf_rag_cache:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF not found at path for Main Chat RAG: {pdf_path}")
                yield None, f"The associated document could not be found."
                return
            pdf_text = _extract_text_from_pdf(pdf_path)
            chunks = _chunk_text(pdf_text)
            # Defensive: if chunking returned no chunks, this could break
            # downstream operations that expect at least one embedding.
            if not chunks:
                logger.error(f"No text chunks extracted from PDF: {pdf_path}")
                yield None, "The associated document could not be processed for context (empty content)."
                return
            embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
            pdf_rag_cache[pdf_path] = {"chunks": chunks, "embeddings": embeddings}
        cached_data = pdf_rag_cache[pdf_path]
        context = _get_top_k_chunks(text, cached_data["chunks"], cached_data["embeddings"])
        rag_history = history[-(HISTORY_LENGTH * 2):]
        if rag_history and isinstance(rag_history[0], dict) and rag_history[0].get('role') == 'assistant':
            rag_history.pop(0)
        
        # Use proper RAG prompt with system instructions
        prompt = f"""{RAG_SYSTEM_PROMPT}

Context from the document:
{context}

Question: {text}

Answer:"""
        
        try:
            response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        except Exception as e:
            if isinstance(e, (httpx.ConnectError, ConnectionRefusedError, OSError)):
                logger.error(f"LLM service connection error for Main Chat RAG (multi): {e}")
                yield None, (
                    "LLM service unreachable. Ensure the local LLM service (e.g. Ollama) is running "
                    "and listening (default: localhost:11434). If you're running in Docker or remote host, update the client URL accordingly."
                )
                return
            logger.error(f"Multi-PDF RAG generation failed: {e}", exc_info=True)
            yield None, f"Error during Multi-PDF RAG completion: {e}"
            return
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
                try:
                    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
                except Exception as e:
                    # If embeddings could not be produced (e.g., empty tensor list), return an informative error
                    logger.error(f"Failed to create embeddings for PDF {pdf_path}: {e}", exc_info=True)
                    yield None, f"Error during Main Chat RAG completion: {e}"
                    return
                pdf_rag_cache[pdf_path] = {"chunks": chunks, "embeddings": embeddings}
                if not chunks:
                    logger.warning(f"No chunks produced for PDF during multi-RAG: {pdf_path}")
                    continue
            cached_data = pdf_rag_cache[pdf_path]
            context = _get_top_k_chunks(text, cached_data["chunks"], cached_data["embeddings"], k=2)
            all_context_chunks.append(f"Context from {os.path.basename(pdf_path)}:\n{context}")
        if not all_context_chunks:
            yield None, "No valid documents could be processed for context."
            return
        combined_context = "\n\n--\n\n".join(all_context_chunks)
        rag_history = history[-(HISTORY_LENGTH * 2):]
        if rag_history and isinstance(rag_history[0], dict) and rag_history[0].get('role') == 'assistant':
            rag_history.pop(0)
        
        # Use proper RAG prompt with system instructions
        prompt = f"""{RAG_SYSTEM_PROMPT}

Context from multiple documents:
{combined_context}

Question: {text}

Answer:"""
        
        response = await asyncio.to_thread(llama3_llm.generate, [prompt])
        answer = response.generations[0][0].text.strip()
        yield answer, None
    except Exception as e:
        logger.error(f"Multi-PDF RAG: {e}", exc_info=True)
        yield None, f"Error during Multi-PDF RAG completion: {e}"

# --- Mindmap Generation using phi3 ---
async def generate_mindmap_from_pdf(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a Mermaid mindmap from PDF using phi3 model.
    Returns: (mindmap_markdown, error_message)
    """
    logger.info(f"Generating mindmap from PDF: {pdf_path}")
    try:
        # Extract text from PDF
        pdf_text = _extract_text_from_pdf(pdf_path)
        
        if not pdf_text or len(pdf_text.strip()) < 50:
            error_msg = "PDF text is too short or empty to generate mindmap"
            logger.error(error_msg)
            return None, error_msg
        
        # Truncate text if too long (keep first 4000 chars for better performance)
        if len(pdf_text) > 4000:
            pdf_text = pdf_text[:4000] + "..."
            logger.info(f"Truncated PDF text to 4000 characters for mindmap generation")
        
        # Generate mindmap using phi3
        prompt = MINDMAP_PROMPT_TEMPLATE.format(pdf_text=pdf_text)
        logger.info("Sending prompt to phi3 for mindmap generation...")
        
        response = await asyncio.to_thread(phi3_llm.generate, [prompt])
        mindmap_markdown = response.generations[0][0].text.strip()
        
        # Validate that we got Mermaid syntax
        if not mindmap_markdown.startswith("mindmap"):
            logger.warning("Generated mindmap doesn't start with 'mindmap', attempting to fix...")
            if "mindmap" in mindmap_markdown.lower():
                # Try to extract the mindmap part
                lines = mindmap_markdown.split('\n')
                mindmap_start = next((i for i, line in enumerate(lines) if line.strip().lower() == "mindmap"), None)
                if mindmap_start is not None:
                    mindmap_markdown = '\n'.join(lines[mindmap_start:])
                else:
                    mindmap_markdown = "mindmap\n" + mindmap_markdown
            else:
                mindmap_markdown = "mindmap\n" + mindmap_markdown
        
        logger.info(f"Mindmap generated successfully ({len(mindmap_markdown)} chars)")
        return mindmap_markdown, None
        
    except Exception as e:
        error_msg = f"Error generating mindmap: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None, error_msg

""" Disabled podcast generation functions as per user request
# --- Mindmap Generation using phi3 ---
# async def generate_mindmap_from_pdf(pdf_path: str) -> Tuple[Optional[str], Optional[str]]:
#     ...
# --- Podcast Generation using phi3 + local TTS ---
# import base64
# import tempfile
# from services.local_audio import generate_tts_audio
# async def generate_podcast_from_mindmap(mindmap_md: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
#     ...
"""

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
SYSTEM_PROMPT  = """You are a helpful AI assistant. Provide clear, accurate, and well-structured answers based on the conversation history. When asked to summarize, organize information logically with bullet points or sections."""

RAG_SYSTEM_PROMPT = """You are an expert document assistant. Analyze the provided context carefully and answer the user's question accurately.

RULES:
1. Use ONLY information from the provided context
2. When summarizing, organize information with clear sections and bullet points
3. Be comprehensive but concise - highlight key points
4. If asked to summarize, structure your response with headings
5. If the answer is not in the context, say "I cannot answer this based on the provided document."
6. Do not copy-paste text directly - synthesize and organize the information
7. Provide well-formatted, easy-to-read responses"""

TITLE_GENERATION_PROMPT = """Based on the following conversation, generate a short, concise title (4-5 words max).
Do not use any quotation marks or labels in your response. Just provide the title text.

CONVERSATION:
{conversation_text}
"""

HISTORY_LENGTH = 10

logger.info("ai_engine.py loaded with local Ollama models: llama3, gemma, phi3.")