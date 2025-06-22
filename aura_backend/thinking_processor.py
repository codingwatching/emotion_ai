"""
Thinking Processor for Aura - Advanced AI Companion (FIXED VERSION)
===================================================================

Implements proper thinking extraction using Google Gemini's thinking capabilities
WITH proper function call handling. This restores the functionality that was
removed while maintaining thinking transparency.

This version properly handles:
- Thinking extraction from Gemini responses
- Function call execution through MCP bridge
- Integration between thinking and tool usage
- Non-streaming mode that works with Aura's architecture
"""

import logging
from typing import List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from google import genai
from google.genai import types
import os
import asyncio
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ThinkingResult:
    """Container for thinking extraction results"""
    thoughts: str
    answer: str
    total_chunks: int
    thinking_chunks: int
    answer_chunks: int
    processing_time_ms: float
    has_thinking: bool
    error: Optional[str] = None

class ThinkingProcessor:
    """
    Advanced thinking processor for Gemini's reasoning capabilities.

    This processor handles responses from Gemini models to extract
    and separate the AI's internal reasoning (thoughts) from the final answer,
    while also properly executing function calls.
    """

    def __init__(self, client: genai.Client):
        """
        Initialize the thinking processor.

        Args:
            client: Initialized Google Gemini client
        """
        self.client = client
        budget = int(os.getenv('THINKING_BUDGET', '-1'))  # Default to adaptive thinking
        self.thinking_budget = budget

    async def process_message_with_thinking(
        self,
        chat: Any,
        message: str,
        user_id: str,
        include_thinking_in_response: bool = False
    ) -> ThinkingResult:
        """
        Process a message with thinking extraction - SIMPLE AND CORRECT VERSION.

        Args:
            chat: Google Gemini chat session with thinking enabled
            message: User's input message
            user_id: User identifier for logging
            include_thinking_in_response: Whether to include thoughts in final answer

        Returns:
            ThinkingResult containing separated thoughts and answer
        """
        start_time = datetime.now()
        thoughts = ""
        answer = ""
        total_chunks = 1
        thinking_chunks = 0
        answer_chunks = 0
        has_thinking = False

        try:
            logger.info(f"🧠 Starting thinking processing for user {user_id}")

            # Use the chat session directly (it should have thinking enabled)
            result = chat.send_message(message)

            # Debug logging
            debug_mode = os.getenv('THINKING_DEBUG', 'true').lower() == 'true'
            if debug_mode:
                logger.debug(f"🔍 Raw response structure for {user_id}: {type(result)}")
                logger.debug(f"🔍 Response candidates: {len(result.candidates) if result.candidates else 0}")

            # Check for valid response
            if not result or not result.candidates:
                raise ValueError("Empty response from Gemini")

            candidate = result.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Malformed response structure from Gemini")

            # Simple, correct processing - trust Gemini's thinking detection
            for i, part in enumerate(candidate.content.parts):
                if debug_mode:
                    has_text = hasattr(part, 'text')
                    has_thought = hasattr(part, 'thought')
                    thought_value = getattr(part, 'thought', None) if has_thought else None
                    logger.debug(f"🔍 Part {i}: type={type(part)}, has_text={has_text}, has_thought={has_thought}, thought_value={thought_value}")

                # Skip parts without text
                if not hasattr(part, 'text') or not part.text:
                    continue

                text_content = str(part.text) if part.text else ""
                if not text_content:
                    continue

                # SIMPLE: Trust Gemini's thinking detection completely
                if hasattr(part, 'thought') and part.thought is True:
                    # This is thinking content as marked by Gemini
                    thoughts += text_content
                    has_thinking = True
                    thinking_chunks += 1
                    logger.info(f"🎯 Gemini thinking - Part {i}: {len(text_content)} chars")
                    if debug_mode:
                        logger.debug(f"🧠 Thinking: {text_content[:200]}...")
                else:
                    # This is regular answer content
                    answer += text_content
                    answer_chunks += 1
                    logger.info(f"💬 Gemini answer - Part {i}: {len(text_content)} chars")
                    if debug_mode:
                        logger.debug(f"💬 Answer: {text_content[:200]}...")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log results
            logger.info(f"✅ Thinking processing complete for user {user_id}")
            logger.info(f"   🧠 Thinking chunks: {thinking_chunks}")
            logger.info(f"   💬 Answer chunks: {answer_chunks}")
            logger.info(f"   🎯 Has thinking: {has_thinking}")
            if has_thinking:
                logger.info(f"   🧠 Thinking length: {len(thoughts)} chars")
            logger.info(f"   💬 Answer length: {len(answer)} chars")

            # Don't include thinking in response unless explicitly requested
            final_answer = answer
            if include_thinking_in_response and has_thinking and thoughts.strip():
                final_answer = f"**My Reasoning:**\n{thoughts}\n\n**My Response:**\n{answer}"

            return ThinkingResult(
                thoughts=thoughts,
                answer=final_answer,
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
                answer="I apologize, but I encountered an issue processing your request. Please try again.",
                total_chunks=0,
                thinking_chunks=0,
                answer_chunks=1,
                processing_time_ms=processing_time,
                has_thinking=False,
                error=str(e)
            )

    async def process_with_function_calls_and_thinking(
        self,
        chat: Any,
        message: str,
        user_id: str,
        mcp_bridge: Any = None,
        include_thinking_in_response: bool = False
    ) -> ThinkingResult:
        """
        Process message with both function calls and enhanced thinking extraction.

        This version preserves the complete reasoning process including tool calls and results
        in the thinking block, as requested by the user.

        Args:
            chat: Google Gemini chat session
            message: User's input message
            user_id: User identifier for logging
            mcp_bridge: MCP bridge for function calls
            include_thinking_in_response: Whether to include thoughts in response

        Returns:
            ThinkingResult with complete thinking process including tool usage
        """
        start_time = datetime.now()

        try:
            logger.info(f"🔧 Starting enhanced thinking+function call processing for user {user_id}")

            # Complete thinking content including tool calls and results
            complete_thinking_process = ""
            final_answer = ""
            total_chunks = 0
            thinking_chunks = 0
            answer_chunks = 0
            has_thinking = False
            function_calls_processed = 0

            # Enable debug mode for thinking processing
            debug_mode = os.getenv('THINKING_DEBUG', 'true').lower() == 'true'

            # Send message to get initial response
            result = chat.send_message(message)

            # Check for empty or malformed response
            if not result or not result.candidates:
                raise ValueError("Empty response from Gemini (possible tool call cutoff)")

            candidate = result.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Malformed response structure from Gemini")

            # Track function calls for complete thinking process
            pending_function_calls = []

            # Process initial response parts
            for i, part in enumerate(candidate.content.parts):
                total_chunks += 1

                if part.text:
                    part_text = part.text
                    
                    # Check if this is thinking content (trust Gemini's marking)
                    if hasattr(part, 'thought') and part.thought is True:
                        complete_thinking_process += part_text + "\n\n"
                        thinking_chunks += 1
                        has_thinking = True
                        logger.info(f"🎯 Initial thinking detected by Gemini attribute - Part {i}")
                        logger.info(f"🧠 Added initial text to thinking (gemini_attribute) - Part {i}: {len(part_text)} chars")
                    else:
                        # This is the final answer content
                        final_answer += part_text
                        answer_chunks += 1
                        logger.info(f"💬 Added initial text to answer - Part {i}: {len(part_text)} chars")

                elif hasattr(part, 'function_call') and part.function_call and mcp_bridge:
                    # Function call detected - add to thinking process
                    pending_function_calls.append(part.function_call)
                    function_calls_processed += 1
                    
                    # Add function call to thinking process
                    complete_thinking_process += f"🔧 **Function Call:** {part.function_call.name}\n"
                    if hasattr(part.function_call, 'args') and part.function_call.args:
                        try:
                            args_str = json.dumps(part.function_call.args, indent=2)
                            complete_thinking_process += f"📋 **Arguments:**\n```json\n{args_str}\n```\n\n"
                        except:
                            complete_thinking_process += f"📋 **Arguments:** {part.function_call.args}\n\n"
                    else:
                        complete_thinking_process += "\n"
                    
                    logger.info(f"🔧 Function call detected: {part.function_call.name}")
                    has_thinking = True

            # Process function calls and add results to thinking
            if pending_function_calls:
                logger.info(f"🔧 Processing {len(pending_function_calls)} function calls")

                for func_call in pending_function_calls:
                    try:
                        # Execute function call through MCP bridge
                        execution_result = await mcp_bridge.execute_function_call(func_call, user_id)

                        if execution_result.success:
                            logger.info(f"✅ Function call {func_call.name} executed successfully")
                            
                            # Add function result to thinking process
                            complete_thinking_process += f"✅ **Function Result for {func_call.name}:**\n"
                            complete_thinking_process += f"```\n{execution_result.result}\n```\n\n"

                            # Send function response back to the model
                            function_response = types.Part(
                                function_response=types.FunctionResponse(
                                    name=func_call.name,
                                    response={"result": execution_result.result}
                                )
                            )

                            # Get follow-up response from the model
                            follow_up_result = chat.send_message([function_response])

                            if (follow_up_result.candidates and
                                follow_up_result.candidates[0].content and
                                follow_up_result.candidates[0].content.parts):

                                # Process follow-up response parts
                                for j, result_part in enumerate(follow_up_result.candidates[0].content.parts):
                                    if result_part.text:
                                        total_chunks += 1
                                        follow_text = result_part.text

                                        # Check if follow-up is thinking or final answer
                                        if hasattr(result_part, 'thought') and result_part.thought is True:
                                            # This is follow-up thinking
                                            complete_thinking_process += f"🧠 **Follow-up Thinking:**\n{follow_text}\n\n"
                                            thinking_chunks += 1
                                            has_thinking = True
                                            logger.info(f"🎯 Follow-up thinking detected (verified with patterns) - Part {j}")
                                            logger.info(f"🧠 Added cleaned follow-up to thinking (gemini_attribute_with_reasoning_patterns) - Part {j}: {len(follow_text)} chars")
                                        else:
                                            # This is the final answer after processing tool results
                                            cleaned_follow_text = self._clean_follow_up_content(follow_text)
                                            if cleaned_follow_text.strip():
                                                final_answer += cleaned_follow_text
                                                answer_chunks += 1
                                                logger.info(f"📝 Follow-up treated as answer - Part {j}")
                                                logger.info(f"💬 Added cleaned follow-up to answer - Part {j}: {len(cleaned_follow_text)} chars")

                        else:
                            # Function call failed - add to thinking process
                            error_info = f"❌ **Function Call Failed:** {func_call.name}\n**Error:** {execution_result.error}\n\n"
                            complete_thinking_process += error_info
                            logger.error(f"❌ Function call {func_call.name} failed: {execution_result.error}")
                            has_thinking = True

                    except Exception as func_error:
                        # Function call error - add to thinking process
                        error_info = f"💥 **Function Call Error:** {func_call.name}\n**Exception:** {str(func_error)}\n\n"
                        complete_thinking_process += error_info
                        logger.error(f"❌ Function call {func_call.name} error: {func_error}")
                        has_thinking = True

            # Ensure we have a response
            if not final_answer.strip():
                if has_thinking and complete_thinking_process.strip():
                    # If we only have thinking content, provide a summary
                    final_answer = "I've completed the analysis and function calls as shown in my reasoning above."
                else:
                    final_answer = "I apologize, but I wasn't able to generate a proper response. Please try again."

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Only include thinking in response if explicitly requested (usually false for UI display)
            response_with_thinking = final_answer
            if include_thinking_in_response and has_thinking and complete_thinking_process.strip():
                response_with_thinking = f"**My Reasoning:**\n{complete_thinking_process}\n\n**My Response:**\n{final_answer}"

            logger.info(f"✅ Enhanced thinking+function processing complete for user {user_id}")
            logger.info(f"   📊 Total chunks: {total_chunks}")
            logger.info(f"   🧠 Thinking chunks: {thinking_chunks}")
            logger.info(f"   💬 Answer chunks: {answer_chunks}")
            logger.info(f"   🔧 Function calls: {function_calls_processed}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.1f}ms")
            logger.info(f"   🎯 Has thinking: {has_thinking}")
            if has_thinking:
                logger.info(f"   🧠 Final thinking length: {len(complete_thinking_process)} chars")
            logger.info(f"   💬 Final answer length: {len(final_answer)} chars")

            return ThinkingResult(
                thoughts=complete_thinking_process,  # Complete process including tool calls and results
                answer=response_with_thinking,       # Clean final answer (without thinking unless requested)
                total_chunks=total_chunks,
                thinking_chunks=thinking_chunks,
                answer_chunks=answer_chunks,
                processing_time_ms=processing_time,
                has_thinking=has_thinking
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Enhanced thinking+function processing failed for user {user_id}: {e}")

            return ThinkingResult(
                thoughts="",
                answer="I apologize, but I encountered an issue processing your request. Please try again.",
                total_chunks=0,
                thinking_chunks=0,
                answer_chunks=1,
                processing_time_ms=processing_time,
                has_thinking=False,
                error=str(e)
            )

    def _clean_follow_up_content(self, content: str) -> str:
        """
        Clean follow-up content to remove conversational headers and unwanted elements.

        This prevents "My Response:", "My Reasoning:", etc. from appearing in the
        AI Reasoning dropdown or main response.

        Args:
            content: Raw follow-up content from Gemini

        Returns:
            Cleaned content with headers and unwanted elements removed
        """
        if not content:
            return ""

        # Remove common conversational headers
        headers_to_remove = [
            "**My Response:**",
            "**My Reasoning:**",
            "**My Analysis:**",
            "**My Thoughts:**",
            "My Response:",
            "My Reasoning:",
            "My Analysis:",
            "My Thoughts:",
            "**Response:**",
            "**Reasoning:**",
            "Response:",
            "Reasoning:"
        ]

        cleaned = content
        for header in headers_to_remove:
            # Remove the header and any following whitespace/newlines
            if header in cleaned:
                parts = cleaned.split(header, 1)
                if len(parts) > 1:
                    # Keep everything before the header, and content after (without the header)
                    cleaned = parts[0] + parts[1]

        # Remove excessive whitespace at start/end
        cleaned = cleaned.strip()

        # Remove leading/trailing newlines that might create formatting issues
        while cleaned.startswith('\n'):
            cleaned = cleaned[1:]
        while cleaned.endswith('\n\n\n'):
            cleaned = cleaned[:-1]

        return cleaned




def create_thinking_enabled_chat(
    client: genai.Client,
    model: str,
    system_instruction: str,
    tools: Optional[List[Any]] = None,
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
        budget = int(os.getenv('THINKING_BUDGET', '-1'))  # Default to adaptive thinking
        thinking_budget = budget

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
