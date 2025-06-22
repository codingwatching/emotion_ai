#!/usr/bin/env python3
"""
Test script to check if responses are being artificially limited
"""

import asyncio
import logging
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_response_limits():
    """Test if responses are being artificially limited"""

    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("❌ GEMINI_API_KEY not found in environment")
        return

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    logger.info("✅ Gemini client initialized")

    # Create a chat session with NO artificial limits
    chat = client.chats.create(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            temperature=0.7,
            # max_output_tokens removed to use default (no artificial limit)
            thinking_config=types.ThinkingConfig(
                thinking_budget=-1,  # Adaptive thinking
                include_thoughts=True
            )
        )
    )
    logger.info("✅ Chat session created with NO LIMITS")

    # Test message that should trigger a longer response
    test_message = "Please give me a comprehensive system check. I want detailed memory statistics, ASEKE framework details, emotional state model information, and any other system information you can provide. Don't hold back - give me everything."

    logger.info(f"🤔 Testing with comprehensive request: {test_message}")

    try:
        # Send message and get response
        result = chat.send_message(test_message)

        # Analyze the response structure
        logger.info("🔍 RESPONSE ANALYSIS:")

        if result.candidates:
            candidate = result.candidates[0]
            if candidate.content and candidate.content.parts:
                total_thinking_chars = 0
                total_answer_chars = 0

                for i, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'text') and part.text:
                        text_content = str(part.text)

                        if hasattr(part, 'thought') and part.thought is True:
                            # Thinking content
                            total_thinking_chars += len(text_content)
                            logger.info(f"   Part {i}: THINKING ({len(text_content)} chars)")
                            logger.info(f"   Thinking preview: {text_content[:200]}...")
                        else:
                            # Answer content
                            total_answer_chars += len(text_content)
                            logger.info(f"   Part {i}: ANSWER ({len(text_content)} chars)")
                            logger.info(f"   Answer preview: {text_content[:200]}...")

                            # Check if answer ends abruptly
                            if text_content.endswith(":") or text_content.endswith("statistics:") or text_content.endswith("..."):
                                logger.warning(f"   ⚠️ Answer appears to be cut off: '{text_content[-50:]}'")

                logger.info(f"📊 TOTALS:")
                logger.info(f"   Total thinking content: {total_thinking_chars} chars")
                logger.info(f"   Total answer content: {total_answer_chars} chars")

                # Check usage metadata
                if hasattr(result, 'usage_metadata') and result.usage_metadata:
                    metadata = result.usage_metadata
                    logger.info(f"   Thoughts tokens: {getattr(metadata, 'thoughts_token_count', 'N/A')}")
                    logger.info(f"   Output tokens: {getattr(metadata, 'candidates_token_count', 'N/A')}")
                    logger.info(f"   Total tokens: {getattr(metadata, 'total_token_count', 'N/A')}")
                else:
                    logger.info("   No usage metadata available")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_response_limits())
