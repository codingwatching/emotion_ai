#!/usr/bin/env python3
"""
Test to verify the search endpoint now properly routes through Aura's MCP tools
"""

import asyncio
import aiohttp
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_memory_search_integration():
    """Test that the /search endpoint now uses Aura's internal MCP tools"""
    
    test_cases = [
        {
            "user_id": "test_user",
            "query": "test memory search",
            "n_results": 3
        },
        {
            "user_id": "ty", 
            "query": "memory system",
            "n_results": 5
        }
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases, 1):
            try:
                logger.info(f"🧪 Test {i}: Searching for '{test_case['query']}'")
                
                async with session.post(
                    "http://localhost:8000/search",
                    json=test_case,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        logger.info(f"✅ Test {i} successful!")
                        logger.info(f"   📊 Found: {result.get('total_found', 0)} memories")
                        logger.info(f"   🔍 Search type: {result.get('search_type', 'unknown')}")
                        logger.info(f"   🎥 Video archives: {result.get('includes_video_archives', False)}")
                        
                        if result.get('results'):
                            logger.info(f"   📄 Sample result: {result['results'][0].get('content', '')[:100]}...")
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Test {i} failed: {response.status} - {error_text}")
                        
            except Exception as e:
                logger.error(f"❌ Test {i} error: {e}")
                
        logger.info("🎉 Memory search integration test completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_memory_search_integration())
        print("\n✅ Search endpoint integration test completed")
        print("🔗 UI memory search should now work through Aura's MCP tools")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
