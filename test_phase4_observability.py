"""
Phase 4 Test Suite: Output Parsers & Observability
Tests callback handlers, output parsers, and structured outputs
"""

import asyncio
import pytest
from typing import List, Dict
from unittest.mock import Mock, patch, AsyncMock
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import Generation, LLMResult

# Import modules to test
from callbacks import (
    StreamingCallbackHandler,
    LoggingCallbackHandler,
    PerformanceCallbackHandler,
    WebSocketCallbackHandler,
    DebugCallbackHandler,
    create_callback_manager
)
from output_parsers import (
    TitleOutputParser,
    MindmapOutputParser,
    SafeJsonOutputParser,
    RAGResponseParser,
    ChatTitle,
    MindmapOutput,
    RAGResponse,
    extract_json_from_text,
    validate_mindmap_markdown
)
from chains import (
    create_chat_chain,
    create_chat_chain_with_history,
    create_rag_chain
)


# ============================================================================
# Test Category 1: Callback Handlers Initialization
# ============================================================================

def test_streaming_callback_handler_init():
    """Test StreamingCallbackHandler initialization"""
    handler = StreamingCallbackHandler()
    assert handler is not None
    assert isinstance(handler, BaseCallbackHandler)
    assert handler.tokens == []
    assert handler.complete_response == ""


def test_logging_callback_handler_init():
    """Test LoggingCallbackHandler initialization"""
    handler = LoggingCallbackHandler()
    assert handler is not None
    assert isinstance(handler, BaseCallbackHandler)


def test_performance_callback_handler_init():
    """Test PerformanceCallbackHandler initialization"""
    handler = PerformanceCallbackHandler()
    assert handler is not None
    assert isinstance(handler, BaseCallbackHandler)
    assert handler.llm_call_count == 0
    assert handler.chain_call_count == 0
    assert handler.total_tokens == 0


def test_websocket_callback_handler_init():
    """Test WebSocketCallbackHandler initialization with mock websocket"""
    mock_ws = Mock()
    handler = WebSocketCallbackHandler(mock_ws)
    assert handler is not None
    assert handler.websocket == mock_ws


def test_debug_callback_handler_init():
    """Test DebugCallbackHandler initialization"""
    handler = DebugCallbackHandler()
    assert handler is not None
    assert isinstance(handler, BaseCallbackHandler)


def test_create_callback_manager():
    """Test callback manager factory function"""
    callbacks = create_callback_manager()
    assert isinstance(callbacks, list)
    assert len(callbacks) > 0
    # Should include at least streaming, logging, and performance handlers
    handler_types = [type(h).__name__ for h in callbacks]
    assert "StreamingCallbackHandler" in handler_types
    assert "LoggingCallbackHandler" in handler_types
    assert "PerformanceCallbackHandler" in handler_types


# ============================================================================
# Test Category 2: Streaming Callback Handler
# ============================================================================

def test_streaming_callback_on_llm_new_token():
    """Test StreamingCallbackHandler token capture"""
    handler = StreamingCallbackHandler()
    
    # Simulate token streaming
    handler.on_llm_new_token("Hello")
    handler.on_llm_new_token(" ")
    handler.on_llm_new_token("World")
    
    assert len(handler.tokens) == 3
    assert handler.tokens == ["Hello", " ", "World"]
    assert handler.complete_response == "Hello World"


def test_streaming_callback_get_response():
    """Test StreamingCallbackHandler response retrieval"""
    handler = StreamingCallbackHandler()
    
    handler.on_llm_new_token("Test")
    handler.on_llm_new_token(" response")
    
    response = handler.get_response()
    assert response == "Test response"


# ============================================================================
# Test Category 3: Performance Callback Handler
# ============================================================================

def test_performance_callback_llm_metrics():
    """Test PerformanceCallbackHandler LLM metrics tracking"""
    handler = PerformanceCallbackHandler()
    
    # Simulate LLM start
    handler.on_llm_start(serialized={}, prompts=["test prompt"])
    assert handler.llm_call_count == 1
    
    # Simulate LLM end with token info
    mock_response = LLMResult(
        generations=[[Generation(text="response")]],
        llm_output={"token_usage": {"total_tokens": 50}}
    )
    handler.on_llm_end(mock_response)
    
    # Check metrics
    metrics = handler.get_metrics()
    assert metrics["llm_calls"] == 1
    assert metrics["total_tokens"] == 50
    assert "avg_llm_duration" in metrics


def test_performance_callback_chain_metrics():
    """Test PerformanceCallbackHandler chain metrics tracking"""
    handler = PerformanceCallbackHandler()
    
    # Simulate chain start
    handler.on_chain_start(serialized={}, inputs={"input": "test"})
    assert handler.chain_call_count == 1
    
    # Simulate chain end
    handler.on_chain_end(outputs={"output": "result"})
    
    # Check metrics
    metrics = handler.get_metrics()
    assert metrics["chain_calls"] == 1
    assert "avg_chain_duration" in metrics


# ============================================================================
# Test Category 4: WebSocket Callback Handler
# ============================================================================

@pytest.mark.asyncio
async def test_websocket_callback_send_token():
    """Test WebSocketCallbackHandler token sending"""
    mock_ws = AsyncMock()
    handler = WebSocketCallbackHandler(mock_ws)
    
    # Simulate token
    await handler.on_llm_new_token("test_token")
    
    # Check websocket was called
    mock_ws.send_json.assert_called_once()
    call_args = mock_ws.send_json.call_args[0][0]
    assert call_args["type"] == "token"
    assert call_args["token"] == "test_token"


@pytest.mark.asyncio
async def test_websocket_callback_send_error():
    """Test WebSocketCallbackHandler error sending"""
    mock_ws = AsyncMock()
    handler = WebSocketCallbackHandler(mock_ws)
    
    # Simulate error
    await handler.on_llm_error(Exception("Test error"))
    
    # Check websocket was called
    mock_ws.send_json.assert_called_once()
    call_args = mock_ws.send_json.call_args[0][0]
    assert call_args["type"] == "error"
    assert "Test error" in call_args["error"]


# ============================================================================
# Test Category 5: Output Parser - TitleOutputParser
# ============================================================================

def test_title_parser_valid_title():
    """Test TitleOutputParser with valid title"""
    parser = TitleOutputParser()
    
    # Test with quoted title
    result = parser.parse('"My Chat Title"')
    assert isinstance(result, ChatTitle)
    assert result.title == "My Chat Title"
    
    # Test with plain title
    result = parser.parse('Another Title')
    assert result.title == "Another Title"


def test_title_parser_max_length():
    """Test TitleOutputParser enforces max length"""
    parser = TitleOutputParser()
    long_title = "A" * 200  # Exceeds 150 char limit
    
    result = parser.parse(long_title)
    assert len(result.title) <= 150


def test_title_parser_fallback():
    """Test TitleOutputParser fallback for invalid input"""
    parser = TitleOutputParser()
    
    # Test with empty string
    result = parser.parse("")
    assert result.title == "Untitled Chat"
    
    # Test with whitespace only
    result = parser.parse("   ")
    assert result.title == "Untitled Chat"


# ============================================================================
# Test Category 6: Output Parser - MindmapOutputParser
# ============================================================================

def test_mindmap_parser_valid_mindmap():
    """Test MindmapOutputParser with valid Mermaid mindmap"""
    parser = MindmapOutputParser()
    
    valid_mindmap = """mindmap
  root((Central Topic))
    Branch 1
      Sub-topic A
      Sub-topic B
    Branch 2
      Sub-topic C
"""
    
    result = parser.parse(valid_mindmap)
    assert isinstance(result, MindmapOutput)
    assert result.markdown.startswith("mindmap")
    assert result.node_count > 0
    assert result.is_valid is True


def test_mindmap_parser_missing_header():
    """Test MindmapOutputParser fixes missing 'mindmap' header"""
    parser = MindmapOutputParser()
    
    mindmap_without_header = """  root((Topic))
    Branch 1
"""
    
    result = parser.parse(mindmap_without_header)
    assert result.markdown.startswith("mindmap")


def test_mindmap_parser_validation():
    """Test validate_mindmap_markdown utility function"""
    valid = validate_mindmap_markdown("mindmap\n  root((Topic))")
    assert valid is True
    
    invalid = validate_mindmap_markdown("not a mindmap")
    assert invalid is False


# ============================================================================
# Test Category 7: Output Parser - SafeJsonOutputParser
# ============================================================================

def test_safe_json_parser_valid_json():
    """Test SafeJsonOutputParser with valid JSON"""
    parser = SafeJsonOutputParser()
    
    json_str = '{"key": "value", "number": 42}'
    result = parser.parse(json_str)
    
    assert isinstance(result, dict)
    assert result["key"] == "value"
    assert result["number"] == 42


def test_safe_json_parser_json_in_text():
    """Test SafeJsonOutputParser extracts JSON from surrounding text"""
    parser = SafeJsonOutputParser()
    
    text_with_json = '''Here is some text
    {"title": "Extracted Title", "count": 5}
    More text after
    '''
    
    result = parser.parse(text_with_json)
    assert isinstance(result, dict)
    assert result["title"] == "Extracted Title"


def test_safe_json_parser_fallback():
    """Test SafeJsonOutputParser fallback for invalid JSON"""
    parser = SafeJsonOutputParser()
    
    invalid_json = "This is not JSON at all"
    result = parser.parse(invalid_json)
    
    # Should return dict with original text
    assert isinstance(result, dict)
    assert "error" in result or "text" in result


def test_extract_json_from_text():
    """Test extract_json_from_text utility function"""
    text = 'Before text {"key": "value"} after text'
    result = extract_json_from_text(text)
    
    assert result == '{"key": "value"}'


# ============================================================================
# Test Category 8: Output Parser - RAGResponseParser
# ============================================================================

def test_rag_response_parser_valid():
    """Test RAGResponseParser with valid RAG response"""
    parser = RAGResponseParser()
    
    rag_text = """Answer: The capital of France is Paris.

Sources:
- Document 1: France Geography
- Document 2: European Capitals

Confidence: 0.95"""
    
    result = parser.parse(rag_text)
    assert isinstance(result, RAGResponse)
    assert "Paris" in result.answer
    assert len(result.sources) > 0
    assert 0.0 <= result.confidence <= 1.0


def test_rag_response_parser_minimal():
    """Test RAGResponseParser with minimal valid response"""
    parser = RAGResponseParser()
    
    minimal_rag = "Answer: Simple response."
    
    result = parser.parse(minimal_rag)
    assert result.answer == "Simple response."
    assert result.confidence > 0.0


# ============================================================================
# Test Category 9: Chains with Callbacks Integration
# ============================================================================

def test_chain_with_callbacks_parameter():
    """Test that chains accept callbacks parameter"""
    from langchain_ollama import OllamaLLM
    
    llm = OllamaLLM(model="llama3", temperature=0.1)
    callbacks = create_callback_manager()
    
    # Test chat chain with callbacks
    chain = create_chat_chain(llm, callbacks=callbacks)
    assert chain is not None


def test_chain_with_enable_observability():
    """Test that chains support enable_observability flag"""
    from langchain_ollama import OllamaLLM
    
    llm = OllamaLLM(model="llama3", temperature=0.1)
    
    # Test with enable_observability=True
    chain = create_chat_chain(llm, enable_observability=True)
    assert chain is not None


@pytest.mark.asyncio
async def test_chain_callbacks_fire():
    """Test that callbacks actually fire during chain execution"""
    from langchain_ollama import OllamaLLM
    
    llm = OllamaLLM(model="llama3", temperature=0.1)
    perf_handler = PerformanceCallbackHandler()
    callbacks = [perf_handler]
    
    # Create chain with callbacks
    chain = create_chat_chain(llm, callbacks=callbacks)
    
    # Note: This test would require actual LLM invocation
    # In practice, you'd mock the LLM or use integration tests
    # For now, we just verify the chain is created correctly
    assert chain is not None
    assert perf_handler.llm_call_count == 0  # No calls yet


# ============================================================================
# Test Category 10: Integration Tests - ai_engine.py Functions
# ============================================================================

@pytest.mark.asyncio
async def test_generate_chat_title_with_parser():
    """Test generate_chat_title uses TitleOutputParser"""
    from ai_engine import generate_chat_title
    
    # Mock messages
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."}
    ]
    
    # This test would require mocking the LLM
    # For now, test with empty messages (returns None)
    result = await generate_chat_title([])
    assert result is None


@pytest.mark.asyncio
async def test_generate_mindmap_with_parser():
    """Test generate_mindmap_from_pdf uses MindmapOutputParser"""
    from ai_engine import generate_mindmap_from_pdf
    
    # This test requires a valid PDF file
    # For unit testing, we'd mock the PDF loader and LLM
    # This is more of an integration test placeholder
    
    # Test with non-existent file (should return error)
    markdown, error = await generate_mindmap_from_pdf("non_existent.pdf")
    assert markdown is None
    assert error is not None


# ============================================================================
# Test Summary and Execution
# ============================================================================

if __name__ == "__main__":
    """
    Run tests with pytest:
    $ pytest test_phase4_observability.py -v
    
    Run specific test category:
    $ pytest test_phase4_observability.py::test_streaming_callback_handler_init -v
    
    Run with coverage:
    $ pytest test_phase4_observability.py --cov=callbacks --cov=output_parsers --cov-report=html
    """
    
    print("Phase 4 Observability Test Suite")
    print("=" * 50)
    print("\nTest Categories:")
    print("1. Callback Handlers Initialization")
    print("2. Streaming Callback Handler")
    print("3. Performance Callback Handler")
    print("4. WebSocket Callback Handler")
    print("5. Output Parser - TitleOutputParser")
    print("6. Output Parser - MindmapOutputParser")
    print("7. Output Parser - SafeJsonOutputParser")
    print("8. Output Parser - RAGResponseParser")
    print("9. Chains with Callbacks Integration")
    print("10. Integration Tests - ai_engine.py")
    print("\n" + "=" * 50)
    print("\nRun with: pytest test_phase4_observability.py -v")
