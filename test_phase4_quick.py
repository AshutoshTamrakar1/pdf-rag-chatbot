"""
Quick unit tests for Phase 4 components (no database required)
"""

import pytest
from callbacks import (
    StreamingCallbackHandler,
    LoggingCallbackHandler,
    PerformanceCallbackHandler,
    DebugCallbackHandler,
    create_callback_manager
)
from output_parsers import (
    TitleOutputParser,
    MindmapOutputParser,
    SafeJsonOutputParser,
    ChatTitle,
    MindmapOutput,
    extract_json_from_text,
    validate_mindmap_markdown
)


# ============================================================================
# Test Callback Handlers
# ============================================================================

def test_streaming_callback_init():
    """Test StreamingCallbackHandler initialization"""
    handler = StreamingCallbackHandler()
    assert handler.tokens == []
    assert handler.complete_response == ""


def test_streaming_callback_tokens():
    """Test StreamingCallbackHandler token capture"""
    handler = StreamingCallbackHandler()
    handler.on_llm_new_token("Hello")
    handler.on_llm_new_token(" World")
    assert len(handler.tokens) == 2
    assert handler.get_response() == "Hello World"


def test_performance_callback_init():
    """Test PerformanceCallbackHandler initialization"""
    handler = PerformanceCallbackHandler()
    assert handler.llm_call_count == 0
    assert handler.chain_call_count == 0
    assert handler.total_tokens == 0


def test_create_callback_manager():
    """Test callback manager factory"""
    callbacks = create_callback_manager()
    assert isinstance(callbacks, list)
    assert len(callbacks) > 0


# ============================================================================
# Test Output Parsers
# ============================================================================

def test_title_parser_valid():
    """Test TitleOutputParser with valid title"""
    parser = TitleOutputParser()
    result = parser.parse('"My Chat Title"')
    assert isinstance(result, ChatTitle)
    assert result.title == "My Chat Title"


def test_title_parser_plain():
    """Test TitleOutputParser with plain title"""
    parser = TitleOutputParser()
    result = parser.parse('Another Title')
    assert result.title == "Another Title"


def test_title_parser_long():
    """Test TitleOutputParser truncates long titles"""
    parser = TitleOutputParser()
    long_title = "A" * 200
    result = parser.parse(long_title)
    assert len(result.title) <= 100


def test_title_parser_empty():
    """Test TitleOutputParser fallback for empty input"""
    parser = TitleOutputParser()
    result = parser.parse("")
    assert result.title == "Untitled Chat"


def test_mindmap_parser_valid():
    """Test MindmapOutputParser with valid mindmap"""
    parser = MindmapOutputParser()
    valid_mindmap = """mindmap
  root((Central Topic))
    Branch 1
      Sub-topic A
    Branch 2
"""
    result = parser.parse(valid_mindmap)
    assert isinstance(result, MindmapOutput)
    assert result.markdown.startswith("mindmap")
    assert result.node_count > 0
    assert result.is_valid is True


def test_mindmap_parser_missing_header():
    """Test MindmapOutputParser fixes missing header"""
    parser = MindmapOutputParser()
    mindmap_without_header = """  root((Topic))
    Branch 1
"""
    result = parser.parse(mindmap_without_header)
    assert result.markdown.startswith("mindmap")


def test_validate_mindmap_markdown():
    """Test validate_mindmap_markdown utility"""
    assert validate_mindmap_markdown("mindmap\n  root((Topic))") is True
    assert validate_mindmap_markdown("not a mindmap") is False


def test_safe_json_parser_valid():
    """Test SafeJsonOutputParser with valid JSON"""
    parser = SafeJsonOutputParser()
    json_str = '{"key": "value", "number": 42}'
    result = parser.parse(json_str)
    assert isinstance(result, dict)
    assert result["key"] == "value"
    assert result["number"] == 42


def test_safe_json_parser_in_text():
    """Test SafeJsonOutputParser extracts JSON from text"""
    parser = SafeJsonOutputParser()
    text = 'Before {"title": "Test", "count": 5} After'
    result = parser.parse(text)
    assert isinstance(result, dict)
    assert result["title"] == "Test"


def test_extract_json_from_text():
    """Test extract_json_from_text utility"""
    text = 'Text {"key": "value"} more text'
    result = extract_json_from_text(text)
    assert result == '{"key": "value"}'


# ============================================================================
# Test Summary
# ============================================================================

if __name__ == "__main__":
    print("Running Phase 4 quick unit tests...")
    pytest.main([__file__, "-v"])
