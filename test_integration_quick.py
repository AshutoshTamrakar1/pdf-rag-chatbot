"""
Quick integration test to verify Phase 3 chain integration with ai_engine.py
This tests that the new LCEL-based functions work correctly.
"""

import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_basic_imports():
    """Test that all imports work."""
    try:
        logger.info("Test 1: Testing imports...")
        from ai_engine import (
            chat_completion_LlamaModel_ws,
            chat_completion_Gemma_ws,
            chat_completion_with_pdf_ws,
            chat_completion_with_multiple_pdfs_ws
        )
        from chains import create_chat_chain, create_rag_chain
        from memory_manager import add_message_to_history, get_windowed_messages
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

async def test_chat_function():
    """Test basic chat function."""
    try:
        logger.info("\nTest 2: Testing basic chat function...")
        from ai_engine import chat_completion_LlamaModel_ws
        
        # Test without session_id (backward compatibility)
        history = []
        async for answer, error in chat_completion_LlamaModel_ws("Say 'Hello'", history):
            if error:
                logger.error(f"❌ Chat test failed: {error}")
                return False
            if answer:
                logger.info(f"✅ Chat test passed. Response: {answer[:100]}...")
                return True
        
        return False
    except Exception as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False

async def test_pdf_rag_function():
    """Test PDF RAG function if PDF available."""
    try:
        logger.info("\nTest 3: Testing PDF RAG function...")
        from ai_engine import chat_completion_with_pdf_ws
        
        # Find a test PDF
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            logger.warning("⚠️  No uploads/ directory found, skipping PDF RAG test")
            return True
        
        pdf_path = None
        for user_dir in uploads_dir.iterdir():
            if user_dir.is_dir():
                for session_dir in user_dir.iterdir():
                    if session_dir.is_dir():
                        for pdf_file in session_dir.glob("*.pdf"):
                            pdf_path = str(pdf_file)
                            break
                    if pdf_path:
                        break
            if pdf_path:
                break
        
        if not pdf_path:
            logger.warning("⚠️  No PDF files found, skipping PDF RAG test")
            return True
        
        logger.info(f"Testing with PDF: {pdf_path}")
        history = []
        async for answer, error in chat_completion_with_pdf_ws(
            "What is this document about?",
            history,
            pdf_path,
            model="llama3"
        ):
            if error:
                logger.error(f"❌ PDF RAG test failed: {error}")
                return False
            if answer:
                logger.info(f"✅ PDF RAG test passed. Response: {answer[:100]}...")
                return True
        
        return False
    except Exception as e:
        logger.error(f"❌ PDF RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_chain_creation():
    """Test that chains can be created directly."""
    try:
        logger.info("\nTest 4: Testing chain creation...")
        from chains import create_chat_chain, get_chain
        from vectorstore_manager import get_vectorstore_manager
        
        # Test basic chat chain
        chat_chain = create_chat_chain("llama3")
        if chat_chain is None:
            logger.error("❌ Failed to create chat chain")
            return False
        
        # Test chain factory
        chat_chain2 = get_chain(chain_type="chat", model_name="llama3")
        if chat_chain2 is None:
            logger.error("❌ Failed to create chat chain via factory")
            return False
        
        logger.info("✅ Chain creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Chain creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_quick_tests():
    """Run all quick integration tests."""
    logger.info("="*70)
    logger.info("QUICK INTEGRATION TEST - Phase 3")
    logger.info("="*70)
    
    results = []
    
    # Test 1: Imports
    results.append(await test_basic_imports())
    
    # Test 2: Chat function
    results.append(await test_chat_function())
    
    # Test 3: Chain creation
    results.append(await test_chain_creation())
    
    # Test 4: PDF RAG (if available)
    results.append(await test_pdf_rag_function())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("\n" + "="*70)
    logger.info(f"RESULTS: {passed}/{total} tests passed")
    logger.info("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_quick_tests())
    sys.exit(0 if success else 1)
