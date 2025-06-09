#!/usr/bin/env python3
"""
Test script to directly verify Aura MCP server parameter handling
===============================================================

This script tests the parameter validation issue by directly calling
the MCP server functions to ensure they work correctly.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add the aura_backend directory to the path
aura_backend_path = Path(__file__).parent / "aura_backend"
sys.path.insert(0, str(aura_backend_path))

async def test_aura_server_tools():
    """Test Aura server tools directly"""
    
    try:
        # Import the MCP tools from aura_server
        from aura_server import search_aura_memories, store_aura_conversation, AuraMemorySearch, AuraConversationStore
        
        print("✅ Successfully imported Aura server tools")
        
        # Test 1: store_aura_conversation with correct parameters
        print("\n" + "="*60)
        print("🧪 Testing store_aura_conversation with correct parameters")
        print("="*60)
        
        store_params = AuraConversationStore(
            user_id="TestUser",
            message="This is a test message",
            sender="user",
            emotional_state="Happy:Medium",
            cognitive_focus="KS"
        )
        
        print(f"📤 Store params: {store_params}")
        
        store_result = await store_aura_conversation(store_params)
        print(f"📥 Store result: {store_result}")
        print("✅ store_aura_conversation test PASSED!")
        
        # Test 2: search_aura_memories with correct parameters
        print("\n" + "="*60)
        print("🧪 Testing search_aura_memories with correct parameters")
        print("="*60)
        
        search_params = AuraMemorySearch(
            user_id="TestUser",
            query="test message",
            n_results=5
        )
        
        print(f"📤 Search params: {search_params}")
        
        search_result = await search_aura_memories(search_params)
        print(f"📥 Search result: {search_result}")
        print("✅ search_aura_memories test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Aura Server Tools Directly")
    print("="*80)
    
    try:
        result = asyncio.run(test_aura_server_tools())
        
        print("\n" + "="*80)
        if result:
            print("🎉 ALL TESTS PASSED! Aura server tools are working correctly.")
        else:
            print("❌ Tests failed. Check the errors above.")
        print("="*80)
        
        sys.exit(0 if result else 1)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
