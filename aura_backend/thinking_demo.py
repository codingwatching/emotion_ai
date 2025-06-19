#!/usr/bin/env python3
"""
Thinking Demo for Aura Backend
=============================

This script demonstrates the thinking extraction capabilities with
interactive examples showing the AI's reasoning process.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging for demo
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def demonstrate_thinking():
    """Interactive demonstration of thinking capabilities"""
    
    try:
        # Import required modules
        from thinking_processor import ThinkingProcessor, create_thinking_enabled_chat
        from google import genai
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY not found in environment variables")
            print("   Please set your Google API key in the .env file")
            return False
        
        print("🚀 Initializing Aura Thinking Demo...")
        
        # Initialize client and processor
        client = genai.Client(api_key=api_key)
        thinking_processor = ThinkingProcessor(client)
        
        # Create thinking-enabled chat
        system_instruction = """You are Aura, an advanced AI companion with transparent reasoning capabilities. 

When solving problems or answering questions:
1. Think through the problem step by step
2. Consider multiple perspectives
3. Show your reasoning process clearly
4. Explain your conclusions

Your thinking process is visible to help users understand how you approach problems."""
        
        chat = create_thinking_enabled_chat(
            client=client,
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            system_instruction=system_instruction,
            thinking_budget=6144  # Generous budget for demo
        )
        
        print("✅ Aura thinking system initialized")
        print("💭 The AI's reasoning process will be captured and displayed")
        print("\n" + "="*70)
        
        # Demo questions that encourage thinking
        demo_questions = [
            {
                "question": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
                "explanation": "This is a classic word problem that tests logical reasoning."
            },
            {
                "question": "What are three creative ways to use a paperclip?",
                "explanation": "This tests creative thinking and problem-solving."
            },
            {
                "question": "If you could travel back in time and change one historical event, what would it be and why?",
                "explanation": "This requires complex reasoning about consequences and ethics."
            }
        ]
        
        for i, demo in enumerate(demo_questions, 1):
            print(f"\n🧪 Demo {i}: {demo['explanation']}")
            print(f"❓ Question: {demo['question']}")
            print("\n⏳ Processing with thinking extraction...")
            
            # Process with thinking extraction
            result = await thinking_processor.process_message_with_thinking(
                chat=chat,
                message=demo['question'],
                user_id="demo_user",
                include_thinking_in_response=False,  # We'll show it separately
                thinking_summary_length=200
            )
            
            # Display thinking process
            print("\n" + "─"*50)
            print("🧠 AURA'S THINKING PROCESS:")
            print("─"*50)
            
            if result.has_thinking:
                # Show thinking summary
                print(f"💭 Summary: {result.thinking_summary}")
                print(f"\n🔍 Detailed Reasoning (first 500 characters):")
                print(f"{result.thoughts[:500]}{'...' if len(result.thoughts) > 500 else ''}")
                
                # Show metrics
                print(f"\n📊 Thinking Metrics:")
                print(f"   • Thinking chunks: {result.thinking_chunks}")
                print(f"   • Answer chunks: {result.answer_chunks}")
                print(f"   • Processing time: {result.processing_time_ms:.1f}ms")
            else:
                print("⚠️ No thinking process captured for this response")
            
            # Display final answer
            print("\n" + "─"*50)
            print("💬 AURA'S RESPONSE:")
            print("─"*50)
            print(result.answer)
            print("\n" + "="*70)
            
            # Pause between demos
            if i < len(demo_questions):
                print("\nPress Enter to continue to the next demo...")
                input()
        
        print("\n🎉 Thinking demonstration completed!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Transparent AI reasoning extraction")
        print("   ✅ Thought summarization")
        print("   ✅ Processing metrics and analysis")
        print("   ✅ Separation of reasoning from final answer")
        
        print("\n⚙️ Configuration Options:")
        print("   • THINKING_BUDGET: Controls reasoning depth (1024-32768 tokens)")
        print("   • INCLUDE_THINKING_IN_RESPONSE: Show reasoning in user responses")
        print("   • Model: Works with Gemini 2.5 models that support thinking")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    print("🌟 Welcome to the Aura Thinking Capabilities Demo!")
    print("=" * 70)
    print("This demo shows how Aura can extract and display the AI's")
    print("reasoning process, making artificial intelligence more transparent.")
    print("=" * 70)
    
    try:
        asyncio.run(demonstrate_thinking())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Thanks for trying the thinking demo!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        sys.exit(1)
