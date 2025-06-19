"""
Thinking Processor for Aura - Advanced AI Companion
==================================================

Implements proper thinking extraction using Google Gemini's thinking capabilities.
This module handles the streaming response processing to separate AI thoughts
from the final answer, enabling transparent reasoning visibility.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from google import genai
from google.genai import types
import os

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ThinkingResult:
    """Container for thinking extraction results"""
    thoughts: str
    answer: str
    thinking_summary: str
    total_chunks: int
    thinking_chunks: int
    answer_chunks: int
    processing_time_ms: float
    has_thinking: bool
    error: Optional[str] = None

class ThinkingProcessor:
    """
    Advanced thinking processor for Gemini's reasoning capabilities.
    
    This processor handles streaming responses from Gemini models to extract
    and separate the AI's internal reasoning (thoughts) from the final answer,
    providing transparency into the AI's decision-making process.
    """
    
    def __init__(self, client: genai.Client):
        """
        Initialize the thinking processor.
        
        Args:
            client: Initialized Google Gemini client
        """
        self.client = client
        self.thinking_budget = int(os.getenv('THINKING_BUDGET', '8192'))
        
    async def process_message_with_thinking(
        self,
        chat: Any,
        message: str,
        user_id: str,
        include_thinking_in_response: bool = False,
        thinking_summary_length: int = 200
    ) -> ThinkingResult:
        """
        Process a message with thinking extraction using streaming.
        
        FIXED: Now uses client.models.generate_content_stream() to properly 
        separate thinking from regular response content.
        
        Args:
            chat: Google Gemini chat session (used for context)
            message: User's input message
            user_id: User identifier for logging
            include_thinking_in_response: Whether to include thoughts in final answer
            thinking_summary_length: Maximum length for thinking summary
            
        Returns:
            ThinkingResult containing separated thoughts and answer
        """
        start_time = datetime.now()
        thoughts = ""
        answer = ""
        total_chunks = 0
        thinking_chunks = 0
        answer_chunks = 0
        has_thinking = False
        
        try:
            logger.info(f"🧠 Starting proper thinking-enabled processing for user {user_id}")
            
            # Build conversation history from chat for context
            conversation_history = []
            
            # Add the current message
            conversation_history.append(message)
            
            # Use the correct method for thinking extraction
            # This is the key fix - using generate_content_stream instead of chat.send_message_stream
            stream = self.client.models.generate_content_stream(
                model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
                contents=conversation_history,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=int(os.getenv('AURA_MAX_OUTPUT_TOKENS', '1000000')),
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_budget=self.thinking_budget
                    )
                )
            )
            
            for chunk in stream:
                total_chunks += 1
                
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                    
                candidate = chunk.candidates[0]
                
                for part in candidate.content.parts:
                    if not part.text:
                        continue
                        
                    # FIXED: Use the correct detection method from Google's example
                    debug_mode = os.getenv('THINKING_DEBUG_MODE', 'false').lower() == 'true'
                    
                    if part.thought:  # Direct check, not hasattr
                        # This is a thought chunk
                        if not has_thinking:
                            logger.info(f"🎯 Thinking detected for user {user_id}")
                            has_thinking = True
                            
                        thoughts += part.text
                        thinking_chunks += 1
                        
                        if debug_mode:
                            logger.info(f"💭 THOUGHT CHUNK {thinking_chunks}: {part.text[:100]}...")
                        else:
                            logger.debug(f"💭 Thought chunk {thinking_chunks}: {part.text[:50]}...")
                    else:
                        # This is regular answer text
                        answer += part.text
                        answer_chunks += 1
                        
                        if debug_mode:
                            logger.info(f"💬 ANSWER CHUNK {answer_chunks}: {part.text[:100]}...")
                        else:
                            logger.debug(f"💬 Answer chunk {answer_chunks}: {part.text[:50]}...")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate thinking summary
            thinking_summary = self._create_thinking_summary(thoughts, thinking_summary_length)
            
            # Log results
            logger.info(f"✅ Thinking processing complete for user {user_id}")
            logger.info(f"   📊 Total chunks: {total_chunks}")
            logger.info(f"   🧠 Thinking chunks: {thinking_chunks}")
            logger.info(f"   💬 Answer chunks: {answer_chunks}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.1f}ms")
            logger.info(f"   🎯 Has thinking: {has_thinking}")
            
            # Log thinking content for debugging
            if has_thinking:
                logger.info(f"   💭 Thinking content preview: {thoughts[:100]}...")
            
            # Optionally include thinking in response
            final_answer = answer
            if include_thinking_in_response and has_thinking and thoughts.strip():
                final_answer = f"**My Reasoning:**\n{thinking_summary}\n\n**My Response:**\n{answer}"
            
            return ThinkingResult(
                thoughts=thoughts,
                answer=final_answer,
                thinking_summary=thinking_summary,
                total_chunks=total_chunks,
                thinking_chunks=thinking_chunks,
                answer_chunks=answer_chunks,
                processing_time_ms=processing_time,
                has_thinking=has_thinking
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Thinking processing failed for user {user_id}: {e}")
            
            return ThinkingResult(
                thoughts="",
                answer=f"I apologize, but I encountered an issue processing your request. Please try again.",
                thinking_summary="",
                total_chunks=total_chunks,
                thinking_chunks=thinking_chunks,
                answer_chunks=answer_chunks,
                processing_time_ms=processing_time,
                has_thinking=False,
                error=str(e)
            )
    
    def _create_thinking_summary(self, thoughts: str, max_length: int = 200) -> str:
        """
        Create a concise summary of the AI's thinking process.
        
        Args:
            thoughts: Raw thinking text
            max_length: Maximum length for summary
            
        Returns:
            Summarized thinking process
        """
        if not thoughts.strip():
            return "No internal reasoning captured."
        
        # Simple summarization - take key sentences
        sentences = thoughts.replace('\n', ' ').split('. ')
        
        if len(thoughts) <= max_length:
            return thoughts.strip()
        
        # Extract key phrases and reasoning steps
        key_phrases = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
                
            # Look for reasoning indicators
            reasoning_indicators = [
                'because', 'therefore', 'since', 'given that',
                'considering', 'based on', 'due to', 'as a result'
            ]
            
            if any(indicator in sentence.lower() for indicator in reasoning_indicators):
                key_phrases.append(sentence)
            elif len(key_phrases) < 2:  # Ensure we have some content
                key_phrases.append(sentence)
        
        summary = '. '.join(key_phrases[:3])  # Max 3 key sentences
        
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary.strip() if summary.strip() else "Complex reasoning process completed."
    
    async def process_with_function_calls_and_thinking(
        self,
        chat: Any,
        message: str,
        user_id: str,
        mcp_bridge: Any = None,
        include_thinking_in_response: bool = False
    ) -> ThinkingResult:
        """
        Process message with both function calls and thinking extraction.
        
        CHALLENGE: Function calls require chat sessions, but thinking extraction works 
        best with generate_content_stream. This tries both approaches intelligently.
        
        Args:
            chat: Google Gemini chat session
            message: User's input message
            user_id: User identifier for logging
            mcp_bridge: MCP bridge for function calls
            include_thinking_in_response: Whether to include thoughts in response
            
        Returns:
            ThinkingResult with function call results integrated
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"🔧 Starting thinking+function call processing for user {user_id}")
            
            # Check if we should force pure thinking extraction (for debugging)
            force_pure_thinking = os.getenv('FORCE_PURE_THINKING_EXTRACTION', 'false').lower() == 'true'
            
            # APPROACH 1: If message seems simple OR forced, use pure thinking extraction
            if force_pure_thinking or not self._message_likely_needs_function_calls(message):
                mode = "forced pure" if force_pure_thinking else "simple message"
                logger.info(f"🧠 Using pure thinking extraction for {user_id} ({mode})")
                return await self.process_message_with_thinking(
                    chat=chat,
                    message=message,
                    user_id=user_id,
                    include_thinking_in_response=include_thinking_in_response
                )
            
            # APPROACH 2: For complex messages, use chat session with improved thinking detection
            logger.info(f"🔧 Message complex, using function-call-aware processing for {user_id}")
            
            thoughts = ""
            answer = ""
            total_chunks = 0
            thinking_chunks = 0
            answer_chunks = 0
            has_thinking = False
            function_calls_processed = 0
            
            # Send message and handle streaming response
            stream = chat.send_message_stream(message)
            
            # Track function calls that need to be processed
            pending_function_calls = []
            
            for chunk in stream:
                total_chunks += 1
                
                if not chunk.candidates or not chunk.candidates[0].content:
                    continue
                    
                candidate = chunk.candidates[0]
                
                for part in candidate.content.parts:
                    if part.text:
                        part_text = part.text
                        
                        # IMPROVED: Better thinking detection for chat streams
                        is_thinking = (
                            # Direct attribute check
                            (hasattr(part, 'thought') and part.thought) or
                            # Content-based heuristics for thinking patterns
                            self._text_appears_to_be_thinking(part_text)
                        )
                        
                        if is_thinking:
                            if not has_thinking:
                                logger.info(f"🎯 Thinking detected in chat stream for user {user_id}")
                                has_thinking = True
                            thoughts += part_text
                            thinking_chunks += 1
                        else:
                            answer += part_text
                            answer_chunks += 1
                            
                    elif hasattr(part, 'function_call') and part.function_call and mcp_bridge:
                        # Handle function calls
                        pending_function_calls.append(part.function_call)
                        function_calls_processed += 1
                        logger.info(f"🔧 Function call detected: {part.function_call.name}")
            
            # Process function calls
            if pending_function_calls:
                for func_call in pending_function_calls:
                    execution_result = await mcp_bridge.execute_function_call(func_call, user_id)
                    
                    if execution_result.success:
                        result_response = chat.send_message([
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=func_call.name,
                                    response={"result": execution_result.result}
                                )
                            )
                        ])
                        
                        if (result_response.candidates and 
                            result_response.candidates[0].content and 
                            result_response.candidates[0].content.parts):
                            
                            for result_part in result_response.candidates[0].content.parts:
                                if result_part.text:
                                    if self._text_appears_to_be_thinking(result_part.text):
                                        thoughts += result_part.text
                                        thinking_chunks += 1
                                    else:
                                        answer += result_part.text
                                        answer_chunks += 1
                    else:
                        answer += f"\n[Function call failed: {execution_result.error}]"
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            thinking_summary = self._create_thinking_summary(thoughts)
            
            final_answer = answer
            if include_thinking_in_response and has_thinking and thoughts.strip():
                final_answer = f"**My Reasoning:**\n{thinking_summary}\n\n**My Response:**\n{answer}"
            
            return ThinkingResult(
                thoughts=thoughts,
                answer=final_answer,
                thinking_summary=thinking_summary,
                total_chunks=total_chunks,
                thinking_chunks=thinking_chunks,
                answer_chunks=answer_chunks,
                processing_time_ms=processing_time,
                has_thinking=has_thinking
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Thinking+function processing failed for user {user_id}: {e}")
            
            return ThinkingResult(
                thoughts="",
                answer="I apologize, but I encountered an issue processing your request. Please try again.",
                thinking_summary="",
                total_chunks=0,
                thinking_chunks=0,
                answer_chunks=0,
                processing_time_ms=processing_time,
                has_thinking=False,
                error=str(e)
            )

    def _message_likely_needs_function_calls(self, message: str) -> bool:
        """Heuristic to determine if a message likely needs function calls"""
        function_indicators = [
            'search', 'find', 'look up', 'query', 'analyze', 'remember',
            'recall', 'get', 'fetch', 'retrieve', 'show me', 'what was',
            'emotional patterns', 'user profile', 'archive', 'export',
            'memories', 'conversation history'
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in function_indicators)
    
    def _text_appears_to_be_thinking(self, text: str) -> bool:
        """Heuristic to detect if text content appears to be thinking/reasoning"""
        thinking_patterns = [
            'let me think', 'i need to', 'considering', 'reflecting on',
            'alright, so', 'now, let\'s see', 'based on', 'given that',
            'my operational state', 'internal processing', 'reasoning',
            'analyzing', 'evaluating', 'processing this'
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in thinking_patterns)

def create_thinking_enabled_chat(
    client: genai.Client,
    model: str,
    system_instruction: str,
    tools: List[Any] = None,
    thinking_budget: Optional[int] = None
) -> Any:
    """
    Create a chat session with thinking capabilities enabled.
    
    Args:
        client: Google Gemini client
        model: Model name to use
        system_instruction: System prompt
        tools: List of available tools
        thinking_budget: Thinking token budget (default from env)
        
    Returns:
        Chat session with thinking enabled
    """
    if thinking_budget is None:
        thinking_budget = int(os.getenv('THINKING_BUDGET', '8192'))
    
    chat = client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=int(os.getenv('AURA_MAX_OUTPUT_TOKENS', '1000000')),
            tools=tools if tools else None,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=thinking_budget
            )
        )
    )
    
    logger.info(f"🧠 Created thinking-enabled chat session (budget: {thinking_budget})")
    return chat
