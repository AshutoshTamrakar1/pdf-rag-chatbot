# LangChain Migration Guide

## Overview
This document tracks the migration from custom RAG implementation to LangChain-powered architecture while maintaining local-only execution.

---

## Migration Status

### ✅ Phase 0: Dependencies & Setup (COMPLETED)
**Date:** December 20, 2025  
**Branch:** `feature/langchain-migration`  
**Tag:** `v1-pre-phase-1`

#### Changes Made:
1. **Updated `requirements.txt`:**
   - Added `langchain-core>=0.1.0`
   - Added `langchain-chroma>=0.1.0`
   - Added `langchain-huggingface>=0.0.1`
   - Added `chromadb>=0.4.22`
   - Added `langchain-mongodb>=0.1.0`
   - Added `langchain-community>=0.0.20`
   - Organized LangChain packages with clear comments

2. **Updated `config.py`:**
   - Added `CHROMA_PERSIST_DIR` setting (default: `./chroma_db`)
   - Added `CHROMA_COLLECTION_NAME` setting (default: `pdf_documents`)
   - Added `EMBEDDING_MODEL_NAME` setting (default: `all-MiniLM-L6-v2`)
   - Added `VECTOR_STORE_SEARCH_TYPE` setting (default: `similarity`)
   - Added `VECTOR_STORE_K` setting (default: `3`)
   - Added `VECTOR_STORE_FETCH_K` setting (default: `10`)
   - Updated directory creation to include Chroma DB path

3. **Created `.env.example`:**
   - Documented all new ChromaDB/vector store configurations
   - Added clear comments and usage examples

4. **Git Setup:**
   - Created feature branch: `feature/langchain-migration`
   - Created safety tag: `v1-pre-phase-1`

#### Next Steps:
Install new dependencies:
```bash
pip install -r requirements.txt
```

---

### ⏳ Phase 1: Foundation Layer (PENDING)
**Target:** Week 1  
**Focus:** Vector Store & Document Loaders

#### Tasks:
- [ ] Create `vectorstore_manager.py`
- [ ] Migrate from manual embeddings to Chroma DB
- [ ] Replace PyMuPDF with `PyMuPDFLoader`
- [ ] Update `ai_engine.py` to use Chroma retriever
- [ ] Create migration script for existing PDFs
- [ ] Test retrieval quality and performance

---

### ⏳ Phase 2: Prompt & Memory Layer (PENDING)
**Target:** Week 2  
**Focus:** Prompt Templates & MongoDB Memory

#### Tasks:
- [ ] Convert `prompts.py` to use `ChatPromptTemplate`
- [ ] Implement `SystemMessagePromptTemplate` + `HumanMessagePromptTemplate`
- [ ] Add `MessagesPlaceholder` for chat history
- [ ] Integrate `MongoDBChatMessageHistory`
- [ ] Replace manual history slicing with memory classes
- [ ] Test conversation persistence

---

### ⏳ Phase 3: Chains & LCEL (PENDING)
**Target:** Week 3  
**Focus:** LangChain Expression Language

#### Tasks:
- [ ] Create `chains.py` module
- [ ] Implement basic LCEL chains for general chat
- [ ] Implement RAG chain with retriever
- [ ] Add `ConversationalRetrievalChain`
- [ ] Replace async generators with LCEL streaming
- [ ] Test end-to-end flows

---

### ⏳ Phase 4: Output Parsers & Observability (PENDING)
**Target:** Week 4  
**Focus:** Structured Outputs & Callbacks

#### Tasks:
- [ ] Create `callbacks.py` module
- [ ] Implement `StrOutputParser` for chat responses
- [ ] Implement `PydanticOutputParser` for structured outputs
- [ ] Add streaming callback handlers
- [ ] Add progress/debug callbacks
- [ ] Test streaming and error handling

---

### ⏳ Phase 5: Integration & Cleanup (PENDING)
**Target:** Week 5  
**Focus:** Service Layer Updates & Migration

#### Tasks:
- [ ] Update `services/chat_service.py`
- [ ] Update `services/pdf_service.py`
- [ ] Update `services/mindmap_service.py`
- [ ] Create data migration scripts
- [ ] Update documentation
- [ ] Remove deprecated code
- [ ] Performance testing

---

## Rollback Instructions

### Quick Rollback to Pre-Migration State
```bash
# Rollback code
git checkout developed
git branch -D feature/langchain-migration

# Rollback dependencies
git checkout developed -- requirements.txt
pip install -r requirements.txt
```

### Rollback to Specific Phase
```bash
# Rollback to before Phase 1
git checkout v1-pre-phase-1

# Rollback to before Phase 2 (when available)
git checkout v1-pre-phase-2
```

---

## Installation Instructions

### Install New Dependencies
```bash
# Activate virtual environment (if using)
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```python
# Test imports
python -c "import chromadb; import langchain; import langchain_chroma; print('✅ All imports successful')"
```

---

## Configuration Updates

### Environment Variables
Copy `.env.example` to `.env` and update values:
```bash
cp .env.example .env
```

Key new settings:
- `CHROMA_PERSIST_DIR`: Where to store vector embeddings (default: `./chroma_db`)
- `EMBEDDING_MODEL_NAME`: HuggingFace model for embeddings (default: `all-MiniLM-L6-v2`)
- `VECTOR_STORE_K`: Number of documents to retrieve (default: `3`)

---

## Testing Strategy

### Unit Tests (Per Phase)
```bash
pytest tests/test_vectorstore.py
pytest tests/test_prompts.py
pytest tests/test_memory.py
pytest tests/test_chains.py
pytest tests/test_parsers.py
```

### Integration Tests
```bash
pytest tests/integration/test_rag_flow.py
pytest tests/integration/test_chat_flow.py
```

### Performance Benchmarks
```bash
python scripts/benchmark_retrieval.py
python scripts/benchmark_chains.py
```

---

## Migration Timeline

| Phase | Start Date | End Date | Status |
|-------|-----------|----------|--------|
| Phase 0 | Dec 20, 2025 | Dec 20, 2025 | ✅ Complete |
| Phase 1 | TBD | TBD | ⏳ Pending |
| Phase 2 | TBD | TBD | ⏳ Pending |
| Phase 3 | TBD | TBD | ⏳ Pending |
| Phase 4 | TBD | TBD | ⏳ Pending |
| Phase 5 | TBD | TBD | ⏳ Pending |

**Estimated Total Duration:** 4-5 weeks

---

## Success Metrics

### Performance Targets
- [ ] Retrieval speed: ≤ 200ms (p95)
- [ ] End-to-end response: ≤ 2s (p95)
- [ ] Memory usage: Stable over 24h
- [ ] Concurrent users: 100+

### Quality Targets
- [ ] Retrieval accuracy: >= baseline
- [ ] Zero data loss in migration
- [ ] All tests passing
- [ ] No feature regressions

### Code Quality
- [ ] Test coverage: >= 80%
- [ ] Type hints: 100%
- [ ] No linting errors
- [ ] Documentation: Complete

---

## Known Issues & Considerations

### Compatibility
- Ensure PyMuPDF version compatibility with langchain-community
- ChromaDB requires SQLite3 (built into Python 3.x)
- Motor and langchain-mongodb may need coordination for async operations

### Performance
- ChromaDB initial indexing may take time for large PDF collections
- Consider batch processing for bulk migrations
- Monitor memory usage during vector store operations

### Data Migration
- Existing PDF cache in memory will be migrated to ChromaDB
- Chat history will be migrated to MongoDBChatMessageHistory format
- Backup database before running migration scripts

---

## Support & References

### Documentation
- [LangChain Docs](https://python.langchain.com/docs/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)

### Key Files
- `requirements.txt`: Python dependencies
- `config.py`: Configuration settings
- `.env.example`: Environment variable template
- This file: Migration tracking

---

## Notes
- All changes maintain local-only execution (no cloud APIs)
- Ollama models remain unchanged (llama3, gemma, phi3)
- Existing MongoDB collections preserved
- Backward compatibility maintained during transition
