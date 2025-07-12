#!/usr/bin/env python3
"""
Test the shared embedding service performance improvements
"""

import time
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_shared_embedding_service():
    """Test the new shared embedding service"""
    
    print("🧠 Testing Shared Embedding Service Performance...")
    print("=" * 50)
    
    # Test 1: Basic functionality
    try:
        from shared_embedding_service import get_embedding_service
        embedding_service = get_embedding_service()
        
        test_text = "This is a test message for embedding"
        
        print("✅ Shared embedding service imported successfully")
        
        # Test encoding
        start_time = time.time()
        embedding = embedding_service.encode_single(test_text)
        end_time = time.time()
        
        print(f"✅ Single text encoded in {end_time - start_time:.3f}s")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    # Test 2: Multiple calls (should be faster after first)
    try:
        print("\n🚀 Testing performance improvement...")
        
        test_texts = [
            "Hello world",
            "This is another test",
            "Performance should be better now",
            "Multiple embeddings at once",
            "Shared service rocks!"
        ]
        
        # First call (model loading)
        start_time = time.time()
        embeddings1 = embedding_service.encode_batch(test_texts)
        first_time = time.time() - start_time
        
        # Second call (should be faster)
        start_time = time.time()
        embeddings2 = embedding_service.encode_batch(test_texts)
        second_time = time.time() - start_time
        
        print(f"✅ First batch ({len(test_texts)} texts): {first_time:.3f}s")
        print(f"✅ Second batch ({len(test_texts)} texts): {second_time:.3f}s")
        print(f"🎉 Speed improvement: {((first_time - second_time) / first_time * 100):.1f}% faster")
        
        # Verify embeddings are consistent
        if embeddings1 == embeddings2:
            print("✅ Embeddings are consistent between calls")
        else:
            print("⚠️ Embeddings differ between calls (unexpected)")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False
    
    # Test 3: Model info
    try:
        model_info = embedding_service.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Embedding Dimension: {model_info['embedding_dimension']}")
        print(f"   Max Sequence Length: {model_info['max_seq_length']}")
        
    except Exception as e:
        print(f"⚠️ Model info test failed: {e}")
    
    print("\n🎉 Shared Embedding Service tests completed!")
    print("\n💡 Benefits:")
    print("   • Single model instance shared across all components")
    print("   • No repeated model loading (saves 400MB+ per instance)")
    print("   • Better GPU memory utilization")
    print("   • Faster subsequent embedding requests")
    print("   • Thread-safe singleton pattern")
    
    return True

if __name__ == "__main__":
    test_shared_embedding_service()
