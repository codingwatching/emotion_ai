#!/usr/bin/env python3
"""
Test Thinking Functionality for Aura Backend
============================================

This script tests the thinking extraction capabilities to ensure proper
integration with the Aura emotion AI system.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_thinking_functionality():
    """Test the thinking processor and integration"""
    
    try:
        # Import required modules
        from thinking_processor import ThinkingProcessor, create_thinking_enabled_chat
        from google import genai
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("❌ GOOGLE_API_KEY not found in environment variables")
            return False
        
        # Initialize client
        client = genai.Client(api_key=api_key)
        logger.info("✅ Google Gemini client initialized")
        
        # Initialize thinking processor
        thinking_processor = ThinkingProcessor(client)
        logger.info("✅ Thinking processor initialized")
        
        # Create thinking-enabled chat
        system_instruction = """You are Aura, a helpful AI assistant. Please think through problems step by step and show your reasoning process."""
        
        chat = create_thinking_enabled_chat(
            client=client,
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            system_instruction=system_instruction,
            thinking_budget=4096
        )
        logger.info("✅ Thinking-enabled chat session created")
        
        # Test messages
        test_messages = [
            "What is 127 * 83? Please think through this step by step.",
            "Explain why the sky appears blue during the day.",
            "What are the pros and cons of renewable energy?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            logger.info(f"\n🧪 Test {i}: {message}")
            
            try:
                # Process message with thinking
                result = await thinking_processor.process_message_with_thinking(
                    chat=chat,
                    message=message,
                    user_id="test_user",
                    include_thinking_in_response=False,
                    thinking_summary_length=150
                )
                
                # Display results
                logger.info(f"✅ Test {i} completed successfully")
                logger.info(f"   🧠 Has thinking: {result.has_thinking}")
                logger.info(f"   📊 Thinking chunks: {result.thinking_chunks}")
                logger.info(f"   💬 Answer chunks: {result.answer_chunks}")
                logger.info(f"   ⏱️ Processing time: {result.processing_time_ms:.1f}ms")
                
                if result.has_thinking:
                    logger.info(f"   💭 Thinking summary: {result.thinking_summary}")
                    logger.info(f"   🧠 Full thoughts (first 200 chars): {result.thoughts[:200]}...")
                
                logger.info(f"   💬 Answer: {result.answer}")
                
                # Brief pause between tests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Test {i} failed: {e}")
                return False
        
        logger.info("\n🎉 All thinking tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Thinking test setup failed: {e}")
        return False

async def test_thinking_status_endpoint():
    """Test the thinking status endpoint"""
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8000/thinking-status') as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✅ Thinking status endpoint working:")
                    logger.info(f"   Status: {data.get('status')}")
                    logger.info(f"   Thinking enabled: {data.get('thinking_configuration', {}).get('thinking_enabled')}")
                    logger.info(f"   Budget: {data.get('thinking_configuration', {}).get('thinking_budget')}")
                    return True
                else:
                    logger.error(f"❌ Status endpoint returned {response.status}")
                    return False
                    
    except Exception as e:
        logger.warning(f"⚠️ Could not test status endpoint (server may not be running): {e}")
        return False

if __name__ == "__main__":
    async def main():
        logger.info("🚀 Starting Aura Thinking Functionality Tests")
        
        # Test 1: Basic thinking functionality
        logger.info("\n📝 Testing thinking processor...")
        thinking_success = await test_thinking_functionality()
        
        # Test 2: Status endpoint (optional - requires running server)
        logger.info("\n📡 Testing thinking status endpoint...")
        endpoint_success = await test_thinking_status_endpoint()
        
        # Summary
        logger.info("\n📊 Test Summary:")
        logger.info(f"   Thinking Processor: {'✅ PASS' if thinking_success else '❌ FAIL'}")
        logger.info(f"   Status Endpoint: {'✅ PASS' if endpoint_success else '⚠️ SKIP'}")
        
        if thinking_success:
            logger.info("\n🎉 Thinking functionality is working correctly!")
            logger.info("💡 Tips:")
            logger.info("   - Set INCLUDE_THINKING_IN_RESPONSE=true to show reasoning in responses")
            logger.info("   - Adjust THINKING_BUDGET to control reasoning depth")
            logger.info("   - Check /thinking-status endpoint for system status")
        else:
            logger.error("\n❌ Thinking functionality has issues. Check the logs above.")
            sys.exit(1)
    
    asyncio.run(main())
