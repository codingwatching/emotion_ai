"""
Quick test to verify Aura backend setup
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")

    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False

    try:
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers imported successfully")
    except ImportError as e:
        print(f"❌ SentenceTransformers import failed: {e}")
        return False

    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False

    try:
        import google.generativeai as genai
        print("✅ Google GenerativeAI imported successfully")
    except ImportError as e:
        print(f"❌ Google GenerativeAI import failed: {e}")
        return False

    return True

async def test_chroma_basic():
    """Test basic ChromaDB functionality"""
    print("\n🗄️ Testing ChromaDB basic functionality...")

    try:
        import chromadb
        from chromadb.config import Settings

        # Create a temporary client
        client = chromadb.Client(Settings(allow_reset=True, is_persistent=False))

        # Create a test collection
        collection = client.create_collection("test_collection")
        print("✅ Created test collection")

        # Add some test data
        collection.add(
            documents=["Hello world", "AI is amazing", "Vector databases are useful"],
            ids=["doc1", "doc2", "doc3"]
        )
        print("✅ Added test documents")

        # Query the collection
        results = collection.query(query_texts=["artificial intelligence"], n_results=2)

        # Check if results and results['documents'] are not None and not empty
        if results and results['documents'] and len(results['documents']) > 0:
            print(f"✅ Query returned {len(results['documents'][0])} results")
        else:
            print("⚠️ Query returned no results or empty documents")
            return False

        return True

    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

async def test_embedding_model():
    """Test sentence transformer loading"""
    print("\n🔤 Testing embedding model...")

    try:
        from sentence_transformers import SentenceTransformer

        # Load a lightweight model for testing
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Loaded embedding model")

        # Test encoding
        text = "This is a test sentence for embedding"
        embedding = model.encode(text)
        print(f"✅ Generated embedding of dimension {len(embedding)}")

        return True

    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

async def test_api_key():
    """Test Google API key"""
    print("\n🔑 Testing Google API key...")

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️ GOOGLE_API_KEY not set in environment")
        return False

    if api_key == "your_actual_api_key_here":
        print("⚠️ GOOGLE_API_KEY is still set to placeholder value")
        return False

    print("✅ Google API key is configured")
    return True

async def main():
    """Run all tests"""
    print("🚀 Aura Backend Quick Test")
    print("=" * 40)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("ChromaDB Functionality", test_chroma_basic),
        ("Embedding Model", test_embedding_model),
        ("API Key Configuration", test_api_key)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 40)
    print("📊 Test Results:")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your Aura backend is ready!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
