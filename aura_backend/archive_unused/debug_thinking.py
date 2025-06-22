#!/usr/bin/env python3
"""
Debug script to test thinking extraction
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from thinking_processor import ThinkingProcessor
from google import genai
import asyncio

async def test_thinking_extraction():
    # Sample response with thinking content
    test_response = """Thinking About "The World Right Now"

Okay, so the user's asking a really big question, "How's the world right now?" That's a fascinating topic, but also incredibly complex! There's just so much that goes into that – from the latest news stories and global events to the economic climate, environmental issues, and even the general mood of people around the world. It's a huge undertaking to try and encapsulate all of that at once.

That's a really big and interesting question, Ty! "How the world is right now" can mean so many things – from global events and news to the general mood, economic situations, or even environmental status.

To give you a meaningful answer, could you tell me what aspects of "the world right now" you're most curious about?"""

    # Initialize client and processor
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    processor = ThinkingProcessor(client)

    print("=== TESTING THINKING EXTRACTION ===")
    print(f"Original response length: {len(test_response)}")
    print(f"Original response preview: {test_response[:100]}...")
    print()

    # Test thinking detection
    detected = processor._detect_thinking_in_response(test_response)
    print(f"Thinking detected: {detected}")
    print()

    # Test thinking extraction
    thoughts, clean_answer = processor._extract_thinking_from_response(test_response)

    print("=== EXTRACTION RESULTS ===")
    print(f"Thoughts length: {len(thoughts)}")
    print(f"Clean answer length: {len(clean_answer)}")
    print()

    print("=== THOUGHTS ===")
    print(f"'{thoughts}'")
    print()

    print("=== CLEAN ANSWER ===")
    print(f"'{clean_answer}'")
    print()

    # Test summary creation
    summary = processor._create_thinking_summary(thoughts, 200)
    print("=== THINKING SUMMARY ===")
    print(f"'{summary}'")

if __name__ == "__main__":
    asyncio.run(test_thinking_extraction())
