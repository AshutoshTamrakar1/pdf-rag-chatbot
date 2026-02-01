"""
Test suite for Phase 3: LCEL Chains implementation.

This module validates:
- Basic chat chains
- Chat chains with history
- RAG chains (single PDF)
- Multi-PDF RAG chains
- Streaming functionality
- Integration with Phase 1 (vectorstore) and Phase 2 (prompts, memory)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from chains import (
    create_chat_chain,
    create_chat_chain_with_history,
    create_rag_chain,
    create_rag_chain_with_history,
    create_multi_pdf_rag_chain,
    create_multi_pdf_rag_chain_with_history,
    get_chain,
    format_docs,
    get_session_history,
    AVAILABLE_MODELS
)
from vectorstore_manager import get_vectorstore_manager
from memory_manager import (
    add_message_to_history,
    get_recent_messages,
    clear_history_cache
)
from prompts import CHAT_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        logger.info(f"✅ PASSED: {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        logger.error(f"❌ FAILED: {test_name} - {error}")
    
    def summary(self):
        total = self.passed + self.failed
        logger.info("\n" + "="*70)
        logger.info(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            logger.info("\nFailed tests:")
            for error in self.errors:
                logger.info(f"  - {error}")
        logger.info("="*70)
        return self.failed == 0


results = TestResults()


def find_test_pdf() -> str:
    """Find a PDF file for testing."""
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        raise FileNotFoundError("uploads/ directory not found")
    
    # Find first PDF
    for user_dir in uploads_dir.iterdir():
        if user_dir.is_dir():
            for session_dir in user_dir.iterdir():
                if session_dir.is_dir():
                    for pdf_file in session_dir.glob("*.pdf"):
                        logger.info(f"Found test PDF: {pdf_file}")
                        return str(pdf_file)
    
    raise FileNotFoundError("No PDF files found in uploads/")


def find_test_collection() -> str:
    """Find an existing ChromaDB collection for testing."""
    vectorstore_mgr = get_vectorstore_manager()
    collections = vectorstore_mgr.list_collections()
    
    if not collections:
        raise ValueError("No collections found in ChromaDB. Run Phase 1 tests first.")
    
    collection = collections[0]
    logger.info(f"Using test collection: {collection}")
    return collection


# ============================================================================
# TEST 1: BASIC IMPORTS AND SETUP
# ============================================================================

def test_imports():
    """Test that all required modules are importable."""
    test_name = "Test 1: Basic imports and setup"
    try:
        # Test chains module imports
        from chains import (
            llama3_llm, gemma_llm, phi3_llm,
            get_llm, format_chat_history, format_docs
        )
        
        # Test that models are initialized
        assert llama3_llm is not None, "llama3_llm not initialized"
        assert gemma_llm is not None, "gemma_llm not initialized"
        assert phi3_llm is not None, "phi3_llm not initialized"
        
        # Test model getter
        llm = get_llm("llama3")
        assert llm is not None, "get_llm failed"
        
        # Test helper functions
        assert callable(format_docs), "format_docs not callable"
        assert callable(format_chat_history), "format_chat_history not callable"
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 2: BASIC CHAT CHAIN
# ============================================================================

async def test_basic_chat_chain():
    """Test basic chat chain without history."""
    test_name = "Test 2: Basic chat chain"
    try:
        # Create chain
        chain = create_chat_chain(model_name="llama3")
        assert chain is not None, "Chain creation failed"
        
        # Test simple invocation
        response = await chain.ainvoke({
            "input": "Say 'Hello, World!' and nothing else.",
            "chat_history": []
        })
        
        assert response is not None, "No response from chain"
        assert isinstance(response, str), f"Response not string: {type(response)}"
        assert len(response) > 0, "Empty response"
        
        logger.info(f"Response: {response[:100]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 3: CHAT CHAIN WITH HISTORY
# ============================================================================

async def test_chat_chain_with_history():
    """Test chat chain with conversation history."""
    test_name = "Test 3: Chat chain with history"
    session_id = "test_session_phase3_chat"
    
    try:
        # Clear any existing history
        clear_history_cache(session_id)
        
        # Add some test history
        await add_message_to_history(session_id, "user", "My favorite color is blue.")
        await add_message_to_history(session_id, "assistant", "Got it! Your favorite color is blue.")
        
        # Create chain with history
        chain = create_chat_chain_with_history(
            model_name="llama3",
            session_id=session_id
        )
        
        # Ask about previous conversation
        response = await chain.ainvoke({
            "input": "What is my favorite color?",
            "session_id": session_id
        })
        
        assert response is not None, "No response from chain"
        assert "blue" in response.lower(), "Chain didn't use conversation history"
        
        logger.info(f"Response with history: {response[:100]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False
    finally:
        # Cleanup
        clear_history_cache(session_id)


# ============================================================================
# TEST 4: RAG CHAIN (Single PDF)
# ============================================================================

async def test_rag_chain():
    """Test RAG chain with single PDF."""
    test_name = "Test 4: RAG chain (single PDF)"
    
    try:
        # Find test collection
        collection_name = find_test_collection()
        
        # Create RAG chain
        chain = create_rag_chain(
            collection_name=collection_name,
            model_name="llama3",
            k=3
        )
        
        # Query the document
        response = await chain.ainvoke("What is this document about? Give a brief summary.")
        
        assert response is not None, "No response from RAG chain"
        assert isinstance(response, str), f"Response not string: {type(response)}"
        assert len(response) > 0, "Empty response"
        
        logger.info(f"RAG response: {response[:200]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 5: RAG CHAIN WITH HISTORY
# ============================================================================

async def test_rag_chain_with_history():
    """Test RAG chain with conversation history."""
    test_name = "Test 5: RAG chain with history"
    session_id = "test_session_phase3_rag"
    
    try:
        # Find test collection
        collection_name = find_test_collection()
        
        # Clear history
        clear_history_cache(session_id)
        
        # Add some context to history
        await add_message_to_history(session_id, "user", "I'm interested in the technical details.")
        await add_message_to_history(session_id, "assistant", "I'll focus on the technical aspects for you.")
        
        # Create RAG chain with history
        chain = create_rag_chain_with_history(
            collection_name=collection_name,
            session_id=session_id,
            model_name="llama3",
            k=3
        )
        
        # Query with reference to history
        response = await chain.ainvoke({
            "input": "Based on what I mentioned, what technical details can you find?"
        })
        
        assert response is not None, "No response from RAG chain with history"
        assert len(response) > 0, "Empty response"
        
        logger.info(f"RAG response with history: {response[:200]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False
    finally:
        clear_history_cache(session_id)


# ============================================================================
# TEST 6: MULTI-PDF RAG CHAIN
# ============================================================================

async def test_multi_pdf_rag_chain():
    """Test multi-PDF RAG chain."""
    test_name = "Test 6: Multi-PDF RAG chain"
    
    try:
        # Get multiple collections
        vectorstore_mgr = get_vectorstore_manager()
        collections = vectorstore_mgr.list_collections()
        
        if len(collections) < 2:
            logger.warning("Not enough collections for multi-PDF test. Skipping.")
            results.add_pass(test_name + " (skipped - need 2+ collections)")
            return True
        
        # Use first 2 collections
        collection_names = collections[:2]
        logger.info(f"Testing with collections: {collection_names}")
        
        # Create multi-PDF chain
        chain = create_multi_pdf_rag_chain(
            collection_names=collection_names,
            model_name="llama3",
            k=2
        )
        
        # Query across documents
        response = await chain.ainvoke("What topics are covered across these documents?")
        
        assert response is not None, "No response from multi-PDF chain"
        assert len(response) > 0, "Empty response"
        
        logger.info(f"Multi-PDF response: {response[:200]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 7: STREAMING SUPPORT
# ============================================================================

async def test_streaming():
    """Test streaming functionality of chains."""
    test_name = "Test 7: Streaming support"
    
    try:
        # Create basic chat chain
        chain = create_chat_chain(model_name="llama3")
        
        # Test streaming
        chunks = []
        async for chunk in chain.astream({
            "input": "Count from 1 to 5.",
            "chat_history": []
        }):
            chunks.append(chunk)
            logger.debug(f"Received chunk: {chunk[:50]}...")
        
        assert len(chunks) > 0, "No chunks received from stream"
        
        # Combine chunks
        full_response = "".join(chunks)
        assert len(full_response) > 0, "Empty combined response"
        
        logger.info(f"Streaming test: Received {len(chunks)} chunks")
        logger.info(f"Full response: {full_response[:100]}...")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 8: CHAIN FACTORY
# ============================================================================

async def test_chain_factory():
    """Test the chain factory function."""
    test_name = "Test 8: Chain factory"
    
    try:
        # Test chat chain creation
        chat_chain = get_chain(
            chain_type="chat",
            model_name="llama3"
        )
        assert chat_chain is not None, "Failed to create chat chain via factory"
        
        # Test RAG chain creation
        collection_name = find_test_collection()
        rag_chain = get_chain(
            chain_type="rag",
            model_name="llama3",
            collection_name=collection_name,
            k=3
        )
        assert rag_chain is not None, "Failed to create RAG chain via factory"
        
        # Test with invalid type
        try:
            invalid_chain = get_chain(chain_type="invalid")
            assert False, "Should have raised ValueError for invalid chain type"
        except ValueError:
            pass  # Expected
        
        logger.info("Chain factory tests passed")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 9: INTEGRATION WITH PHASE 1 & 2
# ============================================================================

async def test_integration():
    """Test integration with Phase 1 (vectorstore) and Phase 2 (prompts, memory)."""
    test_name = "Test 9: Integration with Phase 1 & 2"
    
    try:
        # Test vectorstore integration
        vectorstore_mgr = get_vectorstore_manager()
        collections = vectorstore_mgr.list_collections()
        assert len(collections) > 0, "No collections available (Phase 1 issue)"
        
        # Test retriever creation
        collection_name = collections[0]
        retriever = vectorstore_mgr.as_retriever(
            collection_name=collection_name,
            search_kwargs={'k': 3}
        )
        assert retriever is not None, "Failed to create retriever"
        
        # Test prompt template integration
        from prompts import CHAT_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
        assert CHAT_PROMPT_TEMPLATE is not None, "Chat prompt not available"
        assert RAG_PROMPT_TEMPLATE is not None, "RAG prompt not available"
        
        # Test memory integration
        test_session = "test_integration_phase3"
        await add_message_to_history(test_session, "user", "Test message")
        messages = get_recent_messages(test_session)
        assert len(messages) > 0, "Memory integration failed"
        
        clear_history_cache(test_session)
        
        logger.info("Integration tests passed")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False


# ============================================================================
# TEST 10: END-TO-END FLOW
# ============================================================================

async def test_end_to_end():
    """Test complete end-to-end flow with all components."""
    test_name = "Test 10: End-to-end flow"
    session_id = "test_e2e_phase3"
    
    try:
        # Setup
        clear_history_cache(session_id)
        collection_name = find_test_collection()
        
        # Step 1: Simple chat
        chat_chain = create_chat_chain_with_history("llama3", session_id)
        response1 = await chat_chain.ainvoke({
            "input": "Hello! I want to learn about the document.",
            "session_id": session_id
        })
        assert response1 is not None, "Chat failed"
        await add_message_to_history(session_id, "user", "Hello! I want to learn about the document.")
        await add_message_to_history(session_id, "assistant", response1)
        
        # Step 2: RAG query with history
        rag_chain = create_rag_chain_with_history(collection_name, session_id, "llama3", k=3)
        response2 = await rag_chain.ainvoke({
            "input": "What are the main topics in the document?"
        })
        assert response2 is not None, "RAG query failed"
        
        # Step 3: Follow-up with context
        response3 = await rag_chain.ainvoke({
            "input": "Tell me more about the first topic you mentioned."
        })
        assert response3 is not None, "Follow-up failed"
        
        # Verify history was maintained
        history = get_recent_messages(session_id)
        assert len(history) >= 1, "History not maintained"
        
        logger.info("End-to-end flow completed successfully")
        
        results.add_pass(test_name)
        return True
        
    except Exception as e:
        results.add_fail(test_name, str(e))
        return False
    finally:
        clear_history_cache(session_id)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all Phase 3 tests."""
    logger.info("="*70)
    logger.info("PHASE 3 TEST SUITE: LCEL Chains")
    logger.info("="*70)
    
    # Test 1: Imports
    test_imports()
    
    # Test 2: Basic chat chain
    await test_basic_chat_chain()
    
    # Test 3: Chat with history
    await test_chat_chain_with_history()
    
    # Test 4: RAG chain
    await test_rag_chain()
    
    # Test 5: RAG with history
    await test_rag_chain_with_history()
    
    # Test 6: Multi-PDF RAG
    await test_multi_pdf_rag_chain()
    
    # Test 7: Streaming
    await test_streaming()
    
    # Test 8: Chain factory
    await test_chain_factory()
    
    # Test 9: Integration
    await test_integration()
    
    # Test 10: End-to-end
    await test_end_to_end()
    
    # Summary
    success = results.summary()
    
    return success


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
