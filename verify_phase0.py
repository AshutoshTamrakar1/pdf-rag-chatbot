#!/usr/bin/env python3
"""
Phase 0 Verification Script
Tests that all new LangChain dependencies are properly installed.
"""

import sys
from typing import List, Tuple

def test_imports() -> List[Tuple[str, bool, str]]:
    """Test all new dependency imports."""
    results = []
    
    # Core LangChain
    tests = [
        ("langchain", "LangChain Core"),
        ("langchain_core", "LangChain Core Package"),
        ("langchain_ollama", "LangChain Ollama Integration"),
        ("langchain_community", "LangChain Community Package"),
        ("langchain_text_splitters", "LangChain Text Splitters"),
        
        # Vector Store & Embeddings
        ("chromadb", "ChromaDB"),
        ("langchain_chroma", "LangChain ChromaDB Integration"),
        ("langchain_huggingface", "LangChain HuggingFace Integration"),
        
        # Memory
        ("langchain_mongodb", "LangChain MongoDB Integration"),
    ]
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            results.append((description, True, "✅ Installed"))
        except ImportError as e:
            results.append((description, False, f"❌ Missing: {str(e)}"))
    
    return results


def verify_chromadb_client():
    """Test ChromaDB client creation."""
    try:
        import chromadb
        client = chromadb.Client()
        return True, "✅ ChromaDB client initialized"
    except Exception as e:
        return False, f"❌ ChromaDB client failed: {str(e)}"


def verify_embeddings():
    """Test HuggingFace embeddings wrapper."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Test a simple embedding
        test_text = "This is a test."
        result = embeddings.embed_query(test_text)
        if len(result) > 0:
            return True, f"✅ Embeddings working (dim={len(result)})"
        else:
            return False, "❌ Embeddings returned empty vector"
    except Exception as e:
        return False, f"❌ Embeddings failed: {str(e)}"


def verify_config():
    """Test configuration loading."""
    try:
        from config import get_settings
        settings = get_settings()
        
        # Check new ChromaDB settings
        checks = [
            hasattr(settings, "CHROMA_PERSIST_DIR"),
            hasattr(settings, "CHROMA_COLLECTION_NAME"),
            hasattr(settings, "EMBEDDING_MODEL_NAME"),
            hasattr(settings, "VECTOR_STORE_SEARCH_TYPE"),
            hasattr(settings, "VECTOR_STORE_K"),
            hasattr(settings, "VECTOR_STORE_FETCH_K"),
        ]
        
        if all(checks):
            return True, f"✅ Config loaded with ChromaDB settings"
        else:
            return False, f"❌ Config missing some ChromaDB settings"
    except Exception as e:
        return False, f"❌ Config loading failed: {str(e)}"


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Phase 0 Verification: LangChain Migration Dependencies")
    print("=" * 70)
    print()
    
    # Test imports
    print("📦 Testing Package Imports:")
    print("-" * 70)
    import_results = test_imports()
    all_imported = all(success for _, success, _ in import_results)
    
    for description, success, message in import_results:
        status = "✅" if success else "❌"
        print(f"{status} {description:40} {message}")
    print()
    
    # Test ChromaDB client
    print("🗄️  Testing ChromaDB Client:")
    print("-" * 70)
    chroma_success, chroma_msg = verify_chromadb_client()
    print(chroma_msg)
    print()
    
    # Test embeddings
    print("🧮 Testing HuggingFace Embeddings:")
    print("-" * 70)
    embed_success, embed_msg = verify_embeddings()
    print(embed_msg)
    print()
    
    # Test configuration
    print("⚙️  Testing Configuration:")
    print("-" * 70)
    config_success, config_msg = verify_config()
    print(config_msg)
    print()
    
    # Overall result
    print("=" * 70)
    overall_success = all_imported and chroma_success and embed_success and config_success
    
    if overall_success:
        print("✅ SUCCESS: Phase 0 verification passed!")
        print()
        print("Next steps:")
        print("  1. Review LANGCHAIN_MIGRATION.md for Phase 1 details")
        print("  2. Start implementing vectorstore_manager.py")
        print("  3. Begin migrating ai_engine.py to use ChromaDB")
        return 0
    else:
        print("❌ FAILED: Some dependencies are missing or not working correctly")
        print()
        print("Troubleshooting:")
        print("  1. Reinstall dependencies: pip install -r requirements.txt")
        print("  2. Check Python version (requires 3.10+)")
        print("  3. Try upgrading pip: pip install --upgrade pip")
        return 1


if __name__ == "__main__":
    sys.exit(main())
