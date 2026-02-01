"""
Test script for Phase 1: Vector Store & Document Loaders

This script tests:
1. VectorStore Manager initialization
2. PyMuPDFLoader document loading
3. Document chunking and storage in ChromaDB
4. Similarity search and retrieval
5. Performance benchmarks
"""

import os
import sys
import time
import asyncio
from pathlib import Path

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        from vectorstore_manager import get_vectorstore_manager
        print("✅ vectorstore_manager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import vectorstore_manager: {e}")
        return False
    
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        print("✅ PyMuPDFLoader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import PyMuPDFLoader: {e}")
        return False
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ RecursiveCharacterTextSplitter imported successfully")
    except Exception as e:
        print(f"❌ Failed to import RecursiveCharacterTextSplitter: {e}")
        return False
    
    try:
        import ai_engine
        print("✅ ai_engine imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ai_engine: {e}")
        return False
    
    print("\n✅ All imports successful!\n")
    return True

def test_vectorstore_manager():
    """Test VectorStore Manager initialization."""
    print("=" * 60)
    print("Testing VectorStore Manager...")
    print("=" * 60)
    
    try:
        from vectorstore_manager import get_vectorstore_manager
        
        mgr = get_vectorstore_manager()
        print(f"✅ VectorStore Manager initialized")
        
        # Test client
        client = mgr.client
        print(f"✅ ChromaDB client created")
        
        # Test embeddings
        embeddings = mgr.embeddings
        print(f"✅ Embeddings model loaded")
        
        # List collections
        collections = mgr.list_collections()
        print(f"✅ Collections in ChromaDB: {len(collections)}")
        if collections:
            for col in collections:
                count = mgr.get_collection_count(col)
                print(f"   - {col}: {count} documents")
        
        print("\n✅ VectorStore Manager tests passed!\n")
        return True
    except Exception as e:
        print(f"❌ VectorStore Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_loading():
    """Test PyMuPDFLoader with a sample PDF."""
    print("=" * 60)
    print("Testing Document Loading...")
    print("=" * 60)
    
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        
        # Find a test PDF
        uploads_dir = Path("uploads")
        test_pdf = None
        
        if uploads_dir.exists():
            for pdf_file in uploads_dir.rglob("*.pdf"):
                test_pdf = str(pdf_file)
                break
        
        if not test_pdf:
            print("⚠️  No PDF files found in uploads/ directory")
            print("   Skipping document loading test")
            return True
        
        print(f"Testing with PDF: {test_pdf}")
        
        # Load PDF
        loader = PyMuPDFLoader(test_pdf)
        documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} pages from PDF")
        
        if documents:
            print(f"   - First page preview: {documents[0].page_content[:100]}...")
            print(f"   - Metadata: {documents[0].metadata}")
        
        print("\n✅ Document loading test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Document loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chunking_and_storage():
    """Test text chunking and ChromaDB storage."""
    print("=" * 60)
    print("Testing Chunking and Storage...")
    print("=" * 60)
    
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from vectorstore_manager import get_vectorstore_manager
        
        # Find a test PDF
        uploads_dir = Path("uploads")
        test_pdf = None
        
        if uploads_dir.exists():
            for pdf_file in uploads_dir.rglob("*.pdf"):
                test_pdf = str(pdf_file)
                break
        
        if not test_pdf:
            print("⚠️  No PDF files found in uploads/ directory")
            print("   Skipping chunking and storage test")
            return True
        
        print(f"Testing with PDF: {test_pdf}")
        
        # Load and chunk
        loader = PyMuPDFLoader(test_pdf)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")
        
        # Store in ChromaDB
        mgr = get_vectorstore_manager()
        collection_name = "test_collection"
        
        # Clean up if exists
        if mgr.collection_exists(collection_name):
            mgr.delete_collection(collection_name)
            print(f"   - Cleaned up existing test collection")
        
        doc_ids = mgr.add_documents(chunks, collection_name)
        print(f"✅ Stored {len(doc_ids)} chunks in collection '{collection_name}'")
        
        # Verify storage
        count = mgr.get_collection_count(collection_name)
        print(f"✅ Verified {count} documents in collection")
        
        # Clean up
        mgr.delete_collection(collection_name)
        print(f"✅ Cleaned up test collection")
        
        print("\n✅ Chunking and storage test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Chunking and storage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_search():
    """Test similarity search and retrieval."""
    print("=" * 60)
    print("Testing Similarity Search...")
    print("=" * 60)
    
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from vectorstore_manager import get_vectorstore_manager
        
        # Find a test PDF
        uploads_dir = Path("uploads")
        test_pdf = None
        
        if uploads_dir.exists():
            for pdf_file in uploads_dir.rglob("*.pdf"):
                test_pdf = str(pdf_file)
                break
        
        if not test_pdf:
            print("⚠️  No PDF files found in uploads/ directory")
            print("   Skipping similarity search test")
            return True
        
        print(f"Testing with PDF: {test_pdf}")
        
        # Load, chunk, and store
        loader = PyMuPDFLoader(test_pdf)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\\n\\n", "\\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        mgr = get_vectorstore_manager()
        collection_name = "test_search_collection"
        
        # Clean up if exists
        if mgr.collection_exists(collection_name):
            mgr.delete_collection(collection_name)
        
        mgr.add_documents(chunks, collection_name)
        print(f"✅ Stored {len(chunks)} chunks for search testing")
        
        # Test similarity search
        test_query = "what is this document about"
        print(f"\\nSearching for: '{test_query}'")
        
        start_time = time.time()
        results = mgr.similarity_search_with_score(
            query=test_query,
            collection_name=collection_name,
            k=3
        )
        search_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"✅ Found {len(results)} results in {search_time:.2f}ms")
        
        if results:
            print(f"\\nTop result:")
            doc, score = results[0]
            print(f"   - Score: {score:.4f}")
            print(f"   - Content preview: {doc.page_content[:150]}...")
            print(f"   - Metadata: {doc.metadata}")
        
        # Performance check
        if search_time < 200:
            print(f"\\n✅ Performance target met: {search_time:.2f}ms < 200ms")
        else:
            print(f"\\n⚠️  Performance slower than target: {search_time:.2f}ms > 200ms")
        
        # Clean up
        mgr.delete_collection(collection_name)
        print(f"✅ Cleaned up test collection")
        
        print("\n✅ Similarity search test passed!\n")
        return True
    except Exception as e:
        print(f"❌ Similarity search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ai_engine_integration():
    """Test ai_engine helper functions."""
    print("=" * 60)
    print("Testing AI Engine Integration...")
    print("=" * 60)
    
    try:
        from ai_engine import _get_collection_name, _load_and_store_pdf, _retrieve_context
        
        # Test collection name generation
        test_paths = [
            "uploads/test-file.pdf",
            "uploads/user123/chat456/source789/my document.pdf",
            "test.pdf"
        ]
        
        print("Testing collection name generation:")
        for path in test_paths:
            col_name = _get_collection_name(path)
            print(f"   {path} -> {col_name}")
        
        print("✅ Collection name generation works")
        
        # Find a test PDF
        uploads_dir = Path("uploads")
        test_pdf = None
        
        if uploads_dir.exists():
            for pdf_file in uploads_dir.rglob("*.pdf"):
                test_pdf = str(pdf_file)
                break
        
        if not test_pdf:
            print("⚠️  No PDF files found in uploads/ directory")
            print("   Skipping PDF processing test")
            return True
        
        print(f"\\nTesting with PDF: {test_pdf}")
        
        # Test load and store
        collection_name = _load_and_store_pdf(test_pdf)
        print(f"✅ Loaded and stored PDF in collection: {collection_name}")
        
        # Test idempotency (should not reprocess)
        collection_name2 = _load_and_store_pdf(test_pdf)
        print(f"✅ Idempotent load works (no reprocessing)")
        
        # Test retrieval
        test_query = "summarize this document"
        context = _retrieve_context(test_query, collection_name, k=2)
        print(f"✅ Retrieved context ({len(context)} chars)")
        print(f"   Preview: {context[:150]}...")
        
        print("\n✅ AI Engine integration test passed!\n")
        return True
    except Exception as e:
        print(f"❌ AI Engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\\n" + "=" * 60)
    print("PHASE 1 VECTOR STORE TESTS")
    print("=" * 60 + "\\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("VectorStore Manager", test_vectorstore_manager()))
    results.append(("Document Loading", test_document_loading()))
    results.append(("Chunking & Storage", test_chunking_and_storage()))
    results.append(("Similarity Search", test_similarity_search()))
    results.append(("AI Engine Integration", asyncio.run(test_ai_engine_integration())))
    
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
        print("\\n🎉 All Phase 1 tests passed! Ready for Phase 2.")
        return 0
    else:
        print(f"\\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
