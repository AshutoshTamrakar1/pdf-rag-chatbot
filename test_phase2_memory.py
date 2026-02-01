"""
Test script for Phase 2: Prompt Templates & MongoDB Memory

This script tests:
1. LangChain prompt template creation and formatting
2. MongoDB chat history integration
3. ConversationBufferWindowMemory functionality
4. Memory persistence and window pruning
5. Message synchronization
"""

import asyncio
import sys
from pathlib import Path

def test_prompt_templates():
    """Test LangChain prompt template imports and formatting."""
    print("=" * 60)
    print("Testing Prompt Templates...")
    print("=" * 60)
    
    try:
        from prompts import (
            CHAT_PROMPT_TEMPLATE,
            RAG_PROMPT_TEMPLATE,
            MULTI_PDF_RAG_PROMPT_TEMPLATE,
            TITLE_GENERATION_PROMPT_TEMPLATE,
            HISTORY_LENGTH
        )
        from langchain_core.messages import HumanMessage, AIMessage
        
        print("✅ All prompt templates imported successfully")
        
        # Test CHAT_PROMPT_TEMPLATE
        chat_messages = CHAT_PROMPT_TEMPLATE.format_messages(
            chat_history=[
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!")
            ],
            input="How are you?"
        )
        print(f"✅ CHAT_PROMPT_TEMPLATE formatted: {len(chat_messages)} messages")
        
        # Test RAG_PROMPT_TEMPLATE
        rag_messages = RAG_PROMPT_TEMPLATE.format_messages(
            context="Sample PDF context here",
            chat_history=[],
            input="What does the document say?"
        )
        print(f"✅ RAG_PROMPT_TEMPLATE formatted: {len(rag_messages)} messages")
        
        # Test MULTI_PDF_RAG_PROMPT_TEMPLATE
        multi_messages = MULTI_PDF_RAG_PROMPT_TEMPLATE.format_messages(
            context="Context from multiple PDFs",
            chat_history=[],
            input="Summarize the documents"
        )
        print(f"✅ MULTI_PDF_RAG_PROMPT_TEMPLATE formatted: {len(multi_messages)} messages")
        
        # Test TITLE_GENERATION_PROMPT_TEMPLATE
        title_prompt = TITLE_GENERATION_PROMPT_TEMPLATE.format(
            conversation_text="User: Hello\\nAssistant: Hi!"
        )
        print(f"✅ TITLE_GENERATION_PROMPT_TEMPLATE formatted: {len(title_prompt)} chars")
        
        print(f"✅ HISTORY_LENGTH constant: {HISTORY_LENGTH}")
        
        print("\\n✅ All prompt template tests passed!\\n")
        return True
    except Exception as e:
        print(f"❌ Prompt template test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager_imports():
    """Test memory manager module imports."""
    print("=" * 60)
    print("Testing Memory Manager Imports...")
    print("=" * 60)
    
    try:
        from memory_manager import (
            get_mongodb_chat_history,
            get_windowed_messages,
            clear_history_cache,
            get_recent_messages,
            add_message_to_history
        )
        print("✅ All memory manager functions imported successfully")
        
        print("\\n✅ Memory manager import tests passed!\\n")
        return True
    except Exception as e:
        print(f"❌ Memory manager import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mongodb_chat_history():
    """Test MongoDBChatMessageHistory creation and basic operations."""
    print("=" * 60)
    print("Testing MongoDB Chat History...")
    print("=" * 60)
    
    try:
        from memory_manager import get_mongodb_chat_history, clear_history_cache
        from db_manager import _db_manager
        from config import get_settings
        
        settings = get_settings()
        
        # Connect to MongoDB
        await _db_manager.connect(settings.MONGODB_URI, settings.MONGODB_DB_NAME)
        print("✅ Connected to MongoDB")
        
        # Create test session
        test_session_id = "test_phase2_session"
        
        # Get chat history
        history = get_mongodb_chat_history(test_session_id)
        print(f"✅ Created MongoDBChatMessageHistory for session: {test_session_id}")
        
        # Clear any existing messages
        history.clear()
        print("✅ Cleared existing messages")
        
        # Add messages
        history.add_user_message("Hello, this is a test")
        history.add_ai_message("Hi! I received your test message.")
        history.add_user_message("Can you remember this?")
        history.add_ai_message("Yes, I can remember our conversation.")
        print("✅ Added 4 messages to history")
        
        # Retrieve messages
        messages = history.messages
        print(f"✅ Retrieved {len(messages)} messages from history")
        
        # Verify content
        assert len(messages) == 4, f"Expected 4 messages, got {len(messages)}"
        assert messages[0].content == "Hello, this is a test"
        assert messages[1].content == "Hi! I received your test message."
        print("✅ Message content verification passed")
        
        # Clean up
        history.clear()
        clear_history_cache(test_session_id)
        await _db_manager.close()
        print("✅ Cleaned up test data")
        
        print("\\n✅ MongoDB chat history tests passed!\\n")
        return True
    except Exception as e:
        print(f"❌ MongoDB chat history test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_conversation_memory():
    """Test windowed message history with MongoDB backend."""
    print("=" * 60)
    print("Testing Conversation Memory...")
    print("=" * 60)
    
    try:
        from memory_manager import (
            get_windowed_messages,
            get_mongodb_chat_history,
            clear_history_cache
        )
        from db_manager import _db_manager
        from config import get_settings
        
        settings = get_settings()
        
        # Connect to MongoDB
        await _db_manager.connect(settings.MONGODB_URI, settings.MONGODB_DB_NAME)
        print("✅ Connected to MongoDB")
        
        # Create test session
        test_session_id = "test_phase2_memory"
        
        # Clear any existing data
        history = get_mongodb_chat_history(test_session_id)
        history.clear()
        clear_history_cache(test_session_id)
        
        # Add messages through history
        for i in range(1, 7):  # Add 6 message pairs
            history.add_user_message(f"User message {i}")
            history.add_ai_message(f"AI response {i}")
        print("✅ Added 6 message pairs (12 messages total)")
        
        # Get windowed messages (k=3 means last 3 pairs = 6 messages)
        windowed = get_windowed_messages(test_session_id, k=3)
        print(f"✅ Retrieved windowed messages: {len(windowed)} messages")
        
        # Verify window pruning (should keep only last 3 pairs = 6 messages)
        assert len(windowed) == 6, f"Window should have 6 messages, got {len(windowed)}"
        assert windowed[0].content == "User message 4", f"First windowed message should be 'User message 4', got '{windowed[0].content}'"
        print("✅ Window pruning working correctly")
        
        # Verify persistence by clearing cache and re-fetching
        clear_history_cache(test_session_id)
        windowed2 = get_windowed_messages(test_session_id, k=3)
        print(f"✅ Re-loaded messages from MongoDB: {len(windowed2)} messages")
        
        assert len(windowed2) == len(windowed), "Memory persistence failed"
        print("✅ Memory persistence verified")
        
        # Clean up
        history.clear()
        clear_history_cache(test_session_id)
        await _db_manager.close()
        print("✅ Cleaned up test data")
        
        print("\\n✅ Conversation memory tests passed!\\n")
        return True
    except Exception as e:
        print(f"❌ Conversation memory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_helper_functions():
    """Test memory manager helper functions."""
    print("=" * 60)
    print("Testing Memory Helper Functions...")
    print("=" * 60)
    
    try:
        from memory_manager import (
            add_message_to_history,
            get_recent_messages,
            get_mongodb_chat_history,
            clear_history_cache
        )
        from db_manager import _db_manager
        from config import get_settings
        
        settings = get_settings()
        
        # Connect to MongoDB
        await _db_manager.connect(settings.MONGODB_URI, settings.MONGODB_DB_NAME)
        print("✅ Connected to MongoDB")
        
        # Create test session
        test_session_id = "test_phase2_helpers"
        
        # Clear any existing data
        history = get_mongodb_chat_history(test_session_id)
        history.clear()
        clear_history_cache(test_session_id)
        
        # Test add_message_to_history
        add_message_to_history(test_session_id, "user", "Test user message")
        add_message_to_history(test_session_id, "assistant", "Test assistant response")
        print("✅ Added messages via helper function")
        
        # Test get_recent_messages
        recent = get_recent_messages(test_session_id, limit=10)
        print(f"✅ Retrieved {len(recent)} recent messages")
        
        assert len(recent) == 2, f"Expected 2 messages, got {len(recent)}"
        assert recent[0]['role'] == 'user'
        assert recent[0]['content'] == 'Test user message'
        assert recent[1]['role'] == 'assistant'
        print("✅ Message format and content verified")
        
        # Clean up
        history.clear()
        clear_history_cache(test_session_id)
        await _db_manager.close()
        print("✅ Cleaned up test data")
        
        print("\\n✅ Memory helper function tests passed!\\n")
        return True
    except Exception as e:
        print(f"❌ Memory helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\\n" + "=" * 60)
    print("PHASE 2 PROMPT & MEMORY TESTS")
    print("=" * 60 + "\\n")
    
    results = []
    
    # Synchronous tests
    results.append(("Prompt Templates", test_prompt_templates()))
    results.append(("Memory Manager Imports", test_memory_manager_imports()))
    
    # Async tests
    async def run_async_tests():
        async_results = []
        async_results.append(("MongoDB Chat History", await test_mongodb_chat_history()))
        async_results.append(("Conversation Memory", await test_conversation_memory()))
        async_results.append(("Memory Helper Functions", await test_memory_helper_functions()))
        return async_results
    
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All Phase 2 tests passed! Ready for Phase 3.")
        return 0
    else:
        print(f"\\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
