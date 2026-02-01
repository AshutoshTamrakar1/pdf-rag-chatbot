# Phase 3: LCEL Chains - Quick Start Guide

## ✅ Phase 3 Complete!

All LCEL chains have been implemented and integrated into `ai_engine.py`. This guide provides quick reference for using the new chain-based architecture.

---

## What Changed

### Before Phase 3:
- Manual prompt construction
- Direct LLM calls with `asyncio.to_thread()`
- Manual history management
- No composability

### After Phase 3:
- LCEL declarative chains
- Native streaming with `.astream()`
- Automatic memory integration
- Composable and extensible

---

## Quick Usage Examples

### 1. Basic Chat (No Memory)

```python
from chains import create_chat_chain

chain = create_chat_chain(model_name="llama3")
response = await chain.ainvoke({
    "input": "Hello, how are you?",
    "chat_history": []
})
print(response)
```

### 2. Chat with Memory

```python
from chains import create_chat_chain_with_history

chain = create_chat_chain_with_history(
    model_name="llama3",
    session_id="user_session_123"
)

# First message
response1 = await chain.ainvoke({
    "input": "My name is Alice",
    "session_id": "user_session_123"
})

# Follow-up (remembers context)
response2 = await chain.ainvoke({
    "input": "What's my name?",
    "session_id": "user_session_123"
})
# Response: "Your name is Alice"
```

### 3. Single PDF RAG

```python
from chains import create_rag_chain

chain = create_rag_chain(
    collection_name="my_pdf_collection",
    model_name="llama3",
    k=3  # retrieve top 3 chunks
)

response = await chain.ainvoke("What is this document about?")
print(response)
```

### 4. RAG with Memory

```python
from chains import create_rag_chain_with_history

chain = create_rag_chain_with_history(
    collection_name="my_pdf_collection",
    session_id="user_session_123",
    model_name="llama3",
    k=3
)

# First question
response1 = await chain.ainvoke({
    "input": "What are the main topics?"
})

# Follow-up question (uses conversation history)
response2 = await chain.ainvoke({
    "input": "Tell me more about the first topic"
})
```

### 5. Multi-PDF RAG

```python
from chains import create_multi_pdf_rag_chain_with_history

chain = create_multi_pdf_rag_chain_with_history(
    collection_names=["pdf1_collection", "pdf2_collection", "pdf3_collection"],
    session_id="user_session_123",
    model_name="llama3",
    k=2  # retrieve 2 chunks per document
)

response = await chain.ainvoke({
    "input": "Compare the main ideas across these documents"
})
```

### 6. Using Chain Factory

```python
from chains import get_chain

# Create any chain type with unified interface
chat_chain = get_chain(
    chain_type="chat",
    model_name="llama3",
    session_id="session_123"
)

rag_chain = get_chain(
    chain_type="rag",
    collection_name="my_pdf",
    session_id="session_123",
    model_name="llama3",
    k=5
)

multi_rag_chain = get_chain(
    chain_type="multi_rag",
    collection_names=["pdf1", "pdf2"],
    session_id="session_123",
    model_name="llama3"
)
```

### 7. Streaming Responses

```python
from chains import create_chat_chain

chain = create_chat_chain(model_name="llama3")

# Stream response chunk-by-chunk
async for chunk in chain.astream({
    "input": "Write a short story",
    "chat_history": []
}):
    print(chunk, end="", flush=True)
```

---

## Using Updated ai_engine.py Functions

### Chat Completion (Llama3)

```python
from ai_engine import chat_completion_LlamaModel_ws

# Without memory (backward compatible)
async for answer, error in chat_completion_LlamaModel_ws(
    text="Hello!",
    history=[]
):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Response: {answer}")

# With memory (new feature)
async for answer, error in chat_completion_LlamaModel_ws(
    text="Hello!",
    history=[],
    session_id="user_session_123"
):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Response: {answer}")
```

### RAG with Single PDF

```python
from ai_engine import chat_completion_with_pdf_ws

async for answer, error in chat_completion_with_pdf_ws(
    text="Summarize this document",
    history=[],
    pdf_path="/path/to/document.pdf",
    model="llama3",
    session_id="user_session_123"  # optional
):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Response: {answer}")
```

### RAG with Multiple PDFs

```python
from ai_engine import chat_completion_with_multiple_pdfs_ws

async for answer, error in chat_completion_with_multiple_pdfs_ws(
    text="Compare these documents",
    history=[],
    pdf_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
    session_id="user_session_123"  # optional
):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Response: {answer}")
```

---

## Testing

### Run Full Test Suite

```bash
python test_phase3_chains.py
```

**Tests included:**
1. Basic imports and setup
2. Basic chat chain
3. Chat chain with history
4. RAG chain (single PDF)
5. RAG chain with history
6. Multi-PDF RAG chain
7. Streaming support
8. Chain factory
9. Integration with Phase 1 & 2
10. End-to-end flow

### Run Quick Integration Test

```bash
python test_integration_quick.py
```

**Quick tests:**
1. Import validation
2. Chat function
3. Chain creation
4. PDF RAG (if available)

---

## File Structure

```
pdf-rag-chatbot-v2/
├── chains.py                      # LCEL chain implementations
├── ai_engine.py                   # Updated with LCEL integration
├── test_phase3_chains.py          # Comprehensive test suite
├── test_integration_quick.py      # Quick smoke tests
├── PHASE3_COMPLETION.md           # Detailed completion summary
├── LANGCHAIN_MIGRATION.md         # Migration documentation
└── README_PHASE3.md              # This file
```

---

## Key Features

### ✅ Implemented:
- 8 different chain types (chat, RAG, multi-PDF, with/without memory)
- LCEL pipe operator syntax
- Native streaming support
- MongoDB memory integration
- ChromaDB retriever integration
- Backward compatibility maintained
- Comprehensive test suites

### 🎯 Benefits:
- **Cleaner Code:** Declarative chains vs manual prompt building
- **Composability:** Easy to extend and modify chains
- **Streaming:** Built-in `.astream()` support
- **Memory:** Automatic session-based history
- **Error Handling:** Improved with LCEL's built-in features
- **Observability:** Ready for Phase 4 tracing hooks

---

## Chain Types Reference

| Chain Type | Memory | Documents | Use Case |
|------------|--------|-----------|----------|
| `create_chat_chain` | ❌ | N/A | Simple Q&A |
| `create_chat_chain_with_history` | ✅ | N/A | Conversational chat |
| `create_rag_chain` | ❌ | Single | One-time PDF query |
| `create_rag_chain_with_history` | ✅ | Single | PDF conversation |
| `create_multi_pdf_rag_chain` | ❌ | Multiple | Multi-doc query |
| `create_multi_pdf_rag_chain_with_history` | ✅ | Multiple | Multi-doc conversation |
| `get_chain` (factory) | Both | Both | Unified interface |

---

## Common Patterns

### Pattern 1: Session-Based Chat
```python
session_id = "user_session_123"
chain = create_chat_chain_with_history("llama3", session_id)

# Each message automatically adds to session history
for user_message in conversation:
    response = await chain.ainvoke({"input": user_message, "session_id": session_id})
```

### Pattern 2: Document Analysis
```python
# Load PDF once
collection_name = await load_and_store_pdf("document.pdf")

# Multiple queries with memory
chain = create_rag_chain_with_history(collection_name, "session_id", "llama3")
response1 = await chain.ainvoke({"input": "What is the main topic?"})
response2 = await chain.ainvoke({"input": "Can you elaborate on that?"})
```

### Pattern 3: Batch Processing
```python
chain = create_chat_chain("llama3")

# Process multiple inputs
inputs = [
    {"input": "Question 1", "chat_history": []},
    {"input": "Question 2", "chat_history": []},
    {"input": "Question 3", "chat_history": []}
]

responses = await chain.abatch(inputs)
```

---

## Troubleshooting

### Chain Creation Fails
```python
# Check imports
from chains import create_chat_chain
from vectorstore_manager import get_vectorstore_manager

# Verify ChromaDB collections exist
vectorstore_mgr = get_vectorstore_manager()
collections = vectorstore_mgr.list_collections()
print(f"Available collections: {collections}")
```

### Memory Not Working
```python
# Ensure session_id is provided
chain = create_chat_chain_with_history("llama3", session_id="your_session_id")

# Verify MongoDB connection
from memory_manager import get_recent_messages
messages = get_recent_messages("your_session_id")
print(f"History: {messages}")
```

### Ollama Connection Error
```bash
# Start Ollama service
ollama serve

# Verify models are available
ollama list
ollama pull llama3
ollama pull gemma
```

---

## Next Steps

### Phase 4: Output Parsers & Observability
- Add structured output parsers
- Implement streaming callbacks
- Add tracing/observability hooks
- Performance monitoring

### Future Enhancements:
- Chain result caching
- Dynamic model switching
- Advanced retrieval strategies
- Multi-modal support

---

## Resources

- **Full Documentation:** [LANGCHAIN_MIGRATION.md](LANGCHAIN_MIGRATION.md)
- **Completion Summary:** [PHASE3_COMPLETION.md](PHASE3_COMPLETION.md)
- **LangChain Docs:** https://python.langchain.com/docs/
- **LCEL Guide:** https://python.langchain.com/docs/expression_language/

---

## Support

For issues or questions:
1. Check [LANGCHAIN_MIGRATION.md](LANGCHAIN_MIGRATION.md) for detailed migration info
2. Review [PHASE3_COMPLETION.md](PHASE3_COMPLETION.md) for technical details
3. Run test suites to validate setup
4. Check Ollama service status

---

**Phase 3 Status:** ✅ COMPLETE  
**Date:** December 20, 2025  
**Next:** Phase 4 - Output Parsers & Observability
