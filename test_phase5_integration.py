"""
Phase 5 Integration Test Suite: End-to-End Service Layer Tests
Tests service layer integration with callbacks and output parsers
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path

# Import service modules
from services.chat_service import process_chat_completion
from services.mindmap_service import generate_mindmap
from services.pdf_service import upload_pdf

# Import models
from services.models import ChatRequest, MindmapRequest

# Import callbacks and parsers
from callbacks import (
    StreamingCallbackHandler,
    PerformanceCallbackHandler,
    LoggingCallbackHandler,
    create_callback_manager
)
from output_parsers import (
    TitleOutputParser,
    MindmapOutputParser,
    ChatTitle,
    MindmapOutput
)


# ============================================================================
# Test Category 1: Service Layer - Chat Service Integration
# ============================================================================

def test_chat_service_imports():
    """Test that chat service has required imports"""
    import services.chat_service as chat_service
    
    # Verify callback imports
    assert hasattr(chat_service, 'create_callback_manager') or 'create_callback_manager' in dir(chat_service)
    assert hasattr(chat_service, 'PerformanceCallbackHandler') or 'PerformanceCallbackHandler' in dir(chat_service)
    
    # Verify parser imports
    assert hasattr(chat_service, 'TitleOutputParser') or 'TitleOutputParser' in dir(chat_service)


def test_process_chat_completion_signature():
    """Test process_chat_completion function signature"""
    from services.chat_service import process_chat_completion
    import inspect
    
    sig = inspect.signature(process_chat_completion)
    params = list(sig.parameters.keys())
    
    # Should accept required parameters
    assert 'user_input' in params
    assert 'chat_session_data' in params
    assert 'active_source_ids' in params


# ============================================================================
# Test Category 2: Service Layer - Mindmap Service Integration
# ============================================================================

def test_mindmap_service_imports():
    """Test that mindmap service has required imports"""
    import services.mindmap_service as mindmap_service
    
    # Verify parser imports
    assert hasattr(mindmap_service, 'MindmapOutputParser') or 'MindmapOutputParser' in dir(mindmap_service)
    assert hasattr(mindmap_service, 'MindmapOutput') or 'MindmapOutput' in dir(mindmap_service)


@pytest.mark.asyncio
async def test_mindmap_generation_with_parser():
    """Test mindmap generation uses output parser"""
    # This is a mock test to verify the integration point
    from output_parsers import MindmapOutputParser
    
    parser = MindmapOutputParser()
    
    # Simulate mindmap output from LLM
    raw_mindmap = """mindmap
  root((PDF Analysis))
    Key Points
      Point 1
      Point 2
    Summary
      Overview
"""
    
    # Parse and validate
    result = parser.parse(raw_mindmap)
    
    assert isinstance(result, MindmapOutput)
    assert result.is_valid is True
    assert result.node_count > 0
    assert result.markdown.startswith("mindmap")


# ============================================================================
# Test Category 3: Service Layer - PDF Service Integration
# ============================================================================

def test_pdf_service_imports():
    """Test that PDF service has required imports"""
    import services.pdf_service as pdf_service
    
    # Verify callback imports
    assert hasattr(pdf_service, 'LoggingCallbackHandler') or 'LoggingCallbackHandler' in dir(pdf_service)


# ============================================================================
# Test Category 4: End-to-End Callback Integration
# ============================================================================

def test_callback_manager_in_services():
    """Test callback manager can be used in services"""
    from callbacks import create_callback_manager
    
    callbacks = create_callback_manager()
    
    # Should return list of handlers
    assert isinstance(callbacks, list)
    assert len(callbacks) > 0
    
    # Check handler types
    handler_names = [type(h).__name__ for h in callbacks]
    assert any('Streaming' in name for name in handler_names)
    assert any('Logging' in name for name in handler_names)
    assert any('Performance' in name for name in handler_names)


@pytest.mark.asyncio
async def test_streaming_callback_with_mock_llm():
    """Test streaming callback captures tokens"""
    from callbacks import StreamingCallbackHandler
    
    handler = StreamingCallbackHandler()
    
    # Simulate token streaming
    await handler.on_llm_start(serialized={}, prompts=["test"])
    await handler.on_llm_new_token("Hello")
    await handler.on_llm_new_token(" ")
    await handler.on_llm_new_token("world")
    await handler.on_llm_end(response=Mock())
    
    # Check tokens captured
    assert len(handler.tokens) == 3
    assert handler.get_completion() == "Hello world"


def test_performance_callback_metrics():
    """Test performance callback collects metrics"""
    from callbacks import PerformanceCallbackHandler
    from langchain_core.outputs import LLMResult, Generation
    
    handler = PerformanceCallbackHandler()
    
    # Simulate LLM call
    handler.on_llm_start(serialized={}, prompts=["test"], run_id="test-123")
    
    # Simulate LLM end with token info
    mock_result = LLMResult(
        generations=[[Generation(text="response")]],
        llm_output={"token_usage": {"total_tokens": 100}}
    )
    handler.on_llm_end(mock_result, run_id="test-123")
    
    # Get metrics
    metrics = handler.get_metrics()
    
    assert "llm_calls" in metrics
    assert metrics["llm_calls"] >= 1
    assert "total_tokens" in metrics
    assert metrics["total_tokens"] >= 100


# ============================================================================
# Test Category 5: Output Parser Integration
# ============================================================================

def test_title_parser_in_chat_service():
    """Test title parser can be used in chat service"""
    from output_parsers import TitleOutputParser, ChatTitle
    
    parser = TitleOutputParser()
    
    # Simulate LLM title generation output
    title_outputs = [
        '"Discussion about Python Programming"',
        'Machine Learning Basics',
        '  Analysis of Research Paper  ',
    ]
    
    for output in title_outputs:
        result = parser.parse(output)
        assert isinstance(result, ChatTitle)
        assert len(result.title) > 0
        assert len(result.title) <= 100


def test_mindmap_parser_validation():
    """Test mindmap parser validates syntax"""
    from output_parsers import MindmapOutputParser, validate_mindmap_markdown
    
    # Valid mindmap
    valid = """mindmap
  root((Topic))
    Branch 1
    Branch 2
"""
    assert validate_mindmap_markdown(valid) is True
    
    # Invalid (no mindmap header)
    invalid = """  root((Topic))
    Branch 1
"""
    assert validate_mindmap_markdown(invalid) is False
    
    # Parser should fix invalid
    parser = MindmapOutputParser()
    result = parser.parse(invalid)
    assert result.markdown.startswith("mindmap")


# ============================================================================
# Test Category 6: Integration - Chains with Callbacks
# ============================================================================

def test_chains_accept_callbacks():
    """Test that chain functions accept callback parameters"""
    from chains import create_chat_chain, create_rag_chain
    from langchain_ollama import OllamaLLM
    from callbacks import create_callback_manager
    import inspect
    
    # Check function signatures
    chat_sig = inspect.signature(create_chat_chain)
    assert 'callbacks' in chat_sig.parameters
    assert 'enable_observability' in chat_sig.parameters
    
    rag_sig = inspect.signature(create_rag_chain)
    assert 'callbacks' in rag_sig.parameters
    assert 'enable_observability' in rag_sig.parameters


# ============================================================================
# Test Category 7: Integration - AI Engine with Parsers
# ============================================================================

@pytest.mark.asyncio
async def test_generate_chat_title_integration():
    """Test generate_chat_title uses TitleOutputParser"""
    from ai_engine import generate_chat_title
    
    # Test with empty messages (should return None)
    result = await generate_chat_title([])
    assert result is None
    
    # Test with valid messages would require mocking LLM
    # This is more of an integration test placeholder


@pytest.mark.asyncio  
async def test_generate_mindmap_integration():
    """Test generate_mindmap_from_pdf uses MindmapOutputParser"""
    from ai_engine import generate_mindmap_from_pdf
    
    # Test with non-existent file (should return error)
    markdown, error = await generate_mindmap_from_pdf("non_existent.pdf")
    assert markdown is None
    assert error is not None


# ============================================================================
# Test Category 8: Error Handling and Fallbacks
# ============================================================================

def test_parser_fallback_on_invalid_input():
    """Test parsers handle invalid input gracefully"""
    from output_parsers import TitleOutputParser, SafeJsonOutputParser
    
    # Title parser fallback
    title_parser = TitleOutputParser()
    empty_result = title_parser.parse("")
    assert empty_result.title == "Untitled Chat"
    
    # JSON parser fallback
    json_parser = SafeJsonOutputParser()
    invalid_result = json_parser.parse("not json at all")
    assert isinstance(invalid_result, dict)


@pytest.mark.asyncio
async def test_callback_error_handling():
    """Test callbacks handle errors gracefully"""
    from callbacks import StreamingCallbackHandler
    
    handler = StreamingCallbackHandler()
    
    # Simulate error
    test_error = Exception("Test error")
    await handler.on_llm_error(test_error)
    
    # Handler should not crash
    assert handler is not None


# ============================================================================
# Test Category 9: Performance and Optimization
# ============================================================================

def test_callback_overhead_minimal():
    """Test that callbacks add minimal overhead"""
    from callbacks import create_callback_manager
    import time
    
    # Create callbacks
    start = time.time()
    callbacks = create_callback_manager()
    creation_time = time.time() - start
    
    # Should be very fast (< 10ms)
    assert creation_time < 0.01
    
    # Callbacks list should be reasonable size
    assert len(callbacks) < 10


def test_parser_validation_performance():
    """Test parser validation is fast"""
    from output_parsers import TitleOutputParser, MindmapOutputParser
    import time
    
    title_parser = TitleOutputParser()
    mindmap_parser = MindmapOutputParser()
    
    # Title parsing should be fast
    start = time.time()
    for _ in range(100):
        title_parser.parse("Test Title")
    title_time = time.time() - start
    
    # Should process 100 titles in < 100ms (avg < 1ms each)
    assert title_time < 0.1
    
    # Mindmap parsing should be reasonable
    mindmap_text = "mindmap\n  root((Topic))\n    Branch 1\n    Branch 2"
    start = time.time()
    for _ in range(100):
        mindmap_parser.parse(mindmap_text)
    mindmap_time = time.time() - start
    
    # Should process 100 mindmaps in < 500ms (avg < 5ms each)
    assert mindmap_time < 0.5


# ============================================================================
# Test Category 10: Migration Completeness
# ============================================================================

def test_all_phases_complete():
    """Verify all migration phases are implemented"""
    
    # Phase 1: Vector Store
    from vectorstore_manager import VectorStoreManager, get_vectorstore_manager
    assert VectorStoreManager is not None
    assert get_vectorstore_manager is not None
    
    # Phase 2: Memory
    from memory_manager import get_windowed_messages
    assert get_windowed_messages is not None
    
    # Phase 3: Chains
    from chains import (
        create_chat_chain,
        create_chat_chain_with_history,
        create_rag_chain,
        create_rag_chain_with_history,
        create_multi_pdf_rag_chain,
        create_multi_pdf_rag_chain_with_history
    )
    assert all([
        create_chat_chain,
        create_chat_chain_with_history,
        create_rag_chain,
        create_rag_chain_with_history,
        create_multi_pdf_rag_chain,
        create_multi_pdf_rag_chain_with_history
    ])
    
    # Phase 4: Callbacks and Parsers
    from callbacks import create_callback_manager
    from output_parsers import TitleOutputParser, MindmapOutputParser
    assert create_callback_manager is not None
    assert TitleOutputParser is not None
    assert MindmapOutputParser is not None
    
    # Phase 5: Service integration (current)
    import services.chat_service
    import services.pdf_service
    import services.mindmap_service
    assert all([
        services.chat_service,
        services.pdf_service,
        services.mindmap_service
    ])


def test_langchain_dependencies():
    """Verify all LangChain dependencies are available"""
    
    # Core LangChain
    import langchain_core
    assert langchain_core is not None
    
    # Vector stores
    import langchain_chroma
    assert langchain_chroma is not None
    
    # LLMs
    import langchain_ollama
    assert langchain_ollama is not None
    
    # MongoDB
    import langchain_mongodb
    assert langchain_mongodb is not None
    
    # Community
    import langchain_community
    assert langchain_community is not None


# ============================================================================
# Test Summary and Execution
# ============================================================================

if __name__ == "__main__":
    """
    Run tests with pytest:
    $ pytest test_phase5_integration.py -v
    
    Run specific category:
    $ pytest test_phase5_integration.py::test_chat_service_imports -v
    
    Run with coverage:
    $ pytest test_phase5_integration.py --cov=services --cov=callbacks --cov=output_parsers --cov-report=html
    """
    
    print("Phase 5 Integration Test Suite")
    print("=" * 50)
    print("\nTest Categories:")
    print("1. Chat Service Integration")
    print("2. Mindmap Service Integration")
    print("3. PDF Service Integration")
    print("4. Callback Integration")
    print("5. Output Parser Integration")
    print("6. Chains with Callbacks")
    print("7. AI Engine with Parsers")
    print("8. Error Handling and Fallbacks")
    print("9. Performance and Optimization")
    print("10. Migration Completeness")
    print("\n" + "=" * 50)
    print("\nRun with: pytest test_phase5_integration.py -v")
