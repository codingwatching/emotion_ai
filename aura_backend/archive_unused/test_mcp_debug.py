#!/usr/bin/env python3
"""
Targeted test to understand MCP tool execution issues
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_brave_search_directly():
    """Test the Brave search tool execution directly through MCP endpoint"""
    
    test_data = {
        "tool_name": "brave_web_search",
        "arguments": {
            "query": "Meta AI developer services"
        },
        "user_id": "test_user"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info(f"🔧 Testing direct MCP tool execution: {test_data['tool_name']}")
            
            async with session.post(
                "http://localhost:8000/mcp/execute-tool",
                json=test_data
            ) as response:
                status = response.status
                result = await response.text()
                
                logger.info(f"📊 Response status: {status}")
                
                if status == 200:
                    result_json = json.loads(result)
                    logger.info(f"✅ Tool execution response: {json.dumps(result_json, indent=2)}")
                else:
                    logger.error(f"❌ Tool execution failed: {result}")
                    
        except Exception as e:
            logger.error(f"❌ Direct tool test failed: {e}")
            import traceback
            traceback.print_exc()

async def test_conversation_debug():
    """Test conversation with debug logging to see tool execution flow"""
    
    test_data = {
        "user_id": "debug_user",
        "message": "Please use the brave_web_search tool to find information about Python FastAPI framework",
        "session_id": "debug_session"
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("🧪 Testing conversation with tool call request...")
            
            async with session.post(
                "http://localhost:8000/conversation",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=60)  # Longer timeout for tool execution
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get("response", "")
                    
                    logger.info("📝 Full response:")
                    logger.info(response_text)
                    
                    # Check for error indicators
                    error_phrases = [
                        "small issue",
                        "try again", 
                        "error",
                        "failed",
                        "couldn't",
                        "unable"
                    ]
                    
                    for phrase in error_phrases:
                        if phrase.lower() in response_text.lower():
                            logger.warning(f"⚠️ Detected error indicator: '{phrase}'")
                            
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"❌ Conversation test failed: {e}")
            import traceback
            traceback.print_exc()

async def test_mcp_bridge_status():
    """Check MCP bridge status and available functions"""
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("http://localhost:8000/mcp/bridge-status") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"🌉 MCP Bridge Status: {result.get('status')}")
                    logger.info(f"📊 Available functions: {result.get('available_functions', 0)}")
                    
                    if 'sample_functions' in result:
                        for func in result['sample_functions']:
                            if 'brave' in func.get('name', '').lower():
                                logger.info(f"🔍 Found Brave function: {func['name']}")
                                logger.info(f"   Description: {func.get('description', 'N/A')}")
                                logger.info(f"   Parameters: {json.dumps(func.get('parameters', {}), indent=2)}")
                                
        except Exception as e:
            logger.error(f"❌ Bridge status check failed: {e}")

async def main():
    """Run targeted tests"""
    logger.info("🔬 Starting targeted MCP issue investigation...")
    
    # Test 1: Check bridge status
    await test_mcp_bridge_status()
    
    # Test 2: Direct tool execution
    await test_brave_search_directly()
    
    # Test 3: Conversation with debug
    await test_conversation_debug()
    
    logger.info("\n✅ Investigation completed!")

if __name__ == "__main__":
    asyncio.run(main())
