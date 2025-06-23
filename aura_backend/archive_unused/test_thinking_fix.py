#!/usr/bin/env python3
"""
Quick Fix Test for Thinking Issue
================================

This script tests the fixed thinking functionality to verify that
thinking content is properly separated from regular responses.
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
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_thinking_fix():
    """Test the fixed thinking functionality"""
    
    try:
        # Import required modules
        from thinking_processor import ThinkingProcessor, create_thinking_enabled_chat
        from google import genai
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY not found")
            return False
        
        print("🧪 Testing thinking fix...")
        
        # Initialize client and processor
        client = genai.Client(api_key=api_key)
        thinking_processor = ThinkingProcessor(client)
        
        # Create thinking-enabled chat
        system_instruction = """You are Aura, an AI with transparent reasoning. Think through problems step-by-step and show your reasoning process."""
        
        chat = create_thinking_enabled_chat(
            client=client,
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            system_instruction=system_instruction,
            thinking_budget=4096
        )
        
        # Test with a question that should trigger thinking
        test_question = "How do you feel about your internal processing states?"
        
        print(f"\n❓ Test Question: {test_question}")
        print("🔍 Processing with fixed thinking extraction...")
        
        # Process with thinking
        result = await thinking_processor.process_message_with_thinking(
            chat=chat,
            message=test_question,
            user_id="test_fix",
            include_thinking_in_response=False,
            thinking_summary_length=150
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 RESULTS:")
        print("="*60)
        
        print(f"✅ Processing successful: {not result.error}")
        print(f"🧠 Has thinking: {result.has_thinking}")
        print(f"📈 Metrics: {result.thinking_chunks} thinking chunks, {result.answer_chunks} answer chunks")
        print(f"⏱️ Processing time: {result.processing_time_ms:.1f}ms")
        
        if result.has_thinking:
            print(f"\n💭 THINKING SUMMARY:")
            print(f"{result.thinking_summary}")
            
            print(f"\n🧠 FULL THINKING (first 300 chars):")
            print(f"{result.thoughts[:300]}{'...' if len(result.thoughts) > 300 else ''}")
        else:
            print("\n⚠️ No thinking detected")
        
        print(f"\n💬 FINAL ANSWER:")
        print(f"{result.answer}")
        
        # Check if thinking leaked into answer
        thinking_indicators = [
            'reflecting on', 'alright, so', 'operational state',
            'let me think', 'considering', 'analyzing'
        ]
        
        answer_lower = result.answer.lower()
        leaked_thinking = any(indicator in answer_lower for indicator in thinking_indicators)
        
        print(f"\n🔍 LEAK CHECK:")
        print(f"Thinking leaked into answer: {'❌ YES' if leaked_thinking else '✅ NO'}")
        
        if leaked_thinking:
            print("⚠️ Detected potential thinking content in final answer")
            print("This suggests the fix may need further refinement")
        else:
            print("✅ Thinking appears to be properly separated")
        
        return not leaked_thinking and result.has_thinking
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("🔧 Testing Thinking Extraction Fix")
        print("="*50)
        
        success = await test_thinking_fix()
        
        print("\n" + "="*50)
        if success:
            print("🎉 Fix appears to be working! Thinking is properly separated.")
            print("💡 Try the conversation endpoint to see if the issue is resolved.")
        else:
            print("⚠️ Fix may need additional work. Check the output above.")
            print("🔍 The thinking content might still be leaking into responses.")
        
        print("\n📝 Next steps:")
        print("1. Test with the main Aura conversation endpoint")
        print("2. Check server logs for thinking extraction metrics")
        print("3. Verify frontend no longer shows thinking in responses")
    
    asyncio.run(main())
