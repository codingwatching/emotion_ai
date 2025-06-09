#!/usr/bin/env python3
"""
Test Aura + Memvid Integration
"""

import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memvid_integration():
    """Test the memvid integration"""
    print("🧪 Testing Aura + Memvid Integration")
    print("=" * 50)
    
    try:
        # Test memvid import
        try:
            from memvid import MemvidEncoder, MemvidRetriever
            print("✅ Memvid import successful")
        except ImportError as e:
            print(f"❌ Memvid import failed: {e}")
            return False
        
        # Test hybrid system import
        try:
            from aura_memvid_hybrid import AuraMemvidHybrid
            print("✅ Hybrid system import successful")
        except ImportError as e:
            print(f"❌ Hybrid system import failed: {e}")
            return False
        
        # Initialize hybrid system
        try:
            hybrid = AuraMemvidHybrid()
            print("✅ Hybrid system initialization successful")
        except Exception as e:
            print(f"❌ Hybrid system initialization failed: {e}")
            return False
        
        # Test system stats
        try:
            stats = hybrid.get_system_stats()
            print(f"✅ System stats: {stats}")
        except Exception as e:
            print(f"❌ System stats failed: {e}")
            return False
        
        # Test conversation storage
        try:
            memory_id = hybrid.store_conversation(
                user_id="test_user",
                message="Hello, this is a test message",
                response="Hello! This is Aura's test response",
                emotional_state="Happy",
                cognitive_focus="Learning"
            )
            print(f"✅ Conversation storage successful: {memory_id}")
        except Exception as e:
            print(f"❌ Conversation storage failed: {e}")
            return False
        
        # Test unified search
        try:
            results = hybrid.unified_search("test", "test_user")
            print(f"✅ Unified search successful: {results['total_results']} results")
        except Exception as e:
            print(f"❌ Unified search failed: {e}")
            return False
        
        # Test MCP tools import
        try:
            from aura_memvid_mcp_tools import add_memvid_tools
            print("✅ MCP tools import successful")
        except ImportError as e:
            print(f"❌ MCP tools import failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Integration is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_memvid_integration())
