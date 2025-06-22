#!/usr/bin/env python3
"""
Test script to verify thinking extraction from Google Gemini
"""

import asyncio
import logging
import os
from google import genai
from google.genai import types
from thinking_processor import ThinkingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_thinking_extraction():
    """Test the thinking extraction functionality"""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("❌ GEMINI_API_KEY not found in environment")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    logger.info("✅ Gemini client initialized")

    # Create thinking processor
    processor = ThinkingProcessor(client)
    logger.info("✅ Thinking processor initialized")

    # Test with a complex question that should trigger thinking
    test_message = "Solve this step by step: If a train leaves Chicago at 2 PM traveling at 60 mph towards New York (800 miles away), and another train leaves New York at 3 PM traveling at 80 mph towards Chicago, at what time will they meet?"

    logger.info(f"🤔 Testing with message: {test_message}")

    try:
        # Create a chat session with thinking enabled
        chat = client.chats.create(
            model='gemini-2.5-flash',
            config=types.GenerateContentConfig(
                temperature=0.7,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=4096,
                    include_thoughts=True
                )
            )
        )
        logger.info("✅ Chat session created with thinking enabled")

        # Process the message
        result = await processor.process_message_with_thinking(
            chat=chat,
            message=test_message,
            user_id="test_user"
        )

        # Display results
        logger.info("🔍 THINKING EXTRACTION RESULTS:")
        logger.info(f"   ✅ Has thinking: {result.has_thinking}")
        logger.info(f"   🧠 Thinking content length: {len(result.thoughts)} chars")
        logger.info(f"   💬 Answer content length: {len(result.answer)} chars")
        logger.info(f"   📊 Processing time: {result.processing_time_ms:.1f}ms")

        if result.has_thinking and result.thoughts:
            logger.info("🧠 THINKING CONTENT:")
            print("=" * 60)
            print(result.thoughts)
            print("=" * 60)

        logger.info("💬 ANSWER CONTENT:")
        print("=" * 60)
        print(result.answer)
        print("=" * 60)

        if result.error:
            logger.error(f"❌ Error during processing: {result.error}")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_thinking_extraction())
