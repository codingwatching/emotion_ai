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
        include_thinking_in_response: bool = False,
        thinking_summary_length: int = 200
    ) -> ThinkingResult:
        """
        Process a message with thinking extraction using non-streaming approach.

        Args:
            chat: Google Gemini chat session with thinking enabled
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
        total_chunks = 1  # Non-streaming, so just one "chunk"
        thinking_chunks = 0
        answer_chunks = 1
        has_thinking = False

        try:
            logger.info(f"🧠 Starting thinking processing for user {user_id}")

            # Use the chat session directly (it should have thinking enabled)
            result = chat.send_message(message)

            # Debug logging for raw response (only in debug mode)
            debug_mode = os.getenv('THINKING_DEBUG', 'false').lower() == 'true'
            if debug_mode:
                logger.debug(f"🔍 Raw response structure for {user_id}: {type(result)}")
                logger.debug(f"🔍 Response candidates: {len(result.candidates) if result.candidates else 0}")

            # Check for valid response
            if not result or not result.candidates:
                raise ValueError("Empty response from Gemini")

            candidate = result.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Malformed response structure from Gemini")

            # Debug candidate structure
            if debug_mode:
                logger.debug(f"🔍 Candidate parts count: {len(candidate.content.parts)}")

            # Extract the answer text and check for thinking attributes
            for i, part in enumerate(candidate.content.parts):
                if debug_mode:
                    has_text = hasattr(part, 'text')
                    has_thought = hasattr(part, 'thought')
                    logger.debug(f"🔍 Part {i}: type={type(part)}, has_text={has_text}, has_thought={has_thought}")

                # Skip parts without text
                if not hasattr(part, 'text') or not part.text:
                    continue

                text_content = str(part.text) if part.text else ""
                if not text_content:
                    continue

                # Check if this part is thinking content (based on Google's documentation)
                if hasattr(part, 'thought') and part.thought is True:
                    # This is thinking content - add to thoughts
                    thoughts += text_content
                    has_thinking = True
                    thinking_chunks += 1
                    logger.info(f"🎯 Found thinking content in response part {i}: {len(text_content)} chars")
                    if debug_mode:
                        logger.debug(f"🔍 Thinking content preview: {text_content[:100]}...")
                else:
                    # This is regular answer content (including when part.thought is None or False)
                    answer += text_content
                    if debug_mode:
                        logger.debug(f"🔍 Added answer part: {len(text_content)} chars")

            if debug_mode:
                logger.debug(f"🔍 Total answer length: {len(answer)} chars")
                logger.debug(f"🔍 Total thoughts length: {len(thoughts)} chars")
                logger.debug(f"🔍 Has thinking: {has_thinking}")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Create thinking summary
            thinking_summary = self._create_thinking_summary(thoughts, thinking_summary_length)

            # Log results
            logger.info(f"✅ Thinking processing complete for user {user_id}")
            logger.info(f"   📊 Total chunks: {total_chunks}")
            logger.info(f"   🧠 Thinking chunks: {thinking_chunks}")
            logger.info(f"   💬 Answer chunks: {answer_chunks}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.1f}ms")
            logger.info(f"   🎯 Has thinking: {has_thinking}")
            if has_thinking:
                logger.info(f"   🧠 Thinking content length: {len(thoughts)} chars")
                logger.info(f"   💬 Answer content length: {len(answer)} chars")

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
                answer="I apologize, but I encountered an issue processing your request. Please try again.",
                thinking_summary="",
                total_chunks=total_chunks,
                thinking_chunks=thinking_chunks,
                answer_chunks=answer_chunks,
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
        Process message with both function calls and thinking extraction.

        This is the RESTORED version that properly handles function calls while
        extracting thinking content. This was removed but is essential for tools to work.

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

            thoughts = ""
            answer = ""
            total_chunks = 0
            thinking_chunks = 0
            answer_chunks = 0
            has_thinking = False
            function_calls_processed = 0

            # Send message to get initial response
            result = chat.send_message(message)

            # Check for empty or malformed response
            if not result or not result.candidates:
                raise ValueError("Empty response from Gemini (possible tool call cutoff)")

            candidate = result.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Malformed response structure from Gemini")

            # Process all parts in the response
            pending_function_calls = []

            for part in candidate.content.parts:
                total_chunks += 1

                if part.text:
                    part_text = part.text

                    # Check if this is thinking content
                    is_thinking = hasattr(part, 'thought') and part.thought is True

                    if is_thinking:
                        if not has_thinking:
                            logger.info(f"🎯 Thinking detected in response for user {user_id}")
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

            # Process function calls if any were found
            if pending_function_calls:
                logger.info(f"🔧 Processing {len(pending_function_calls)} function calls")

                for func_call in pending_function_calls:
                    try:
                        # Execute function call through MCP bridge
                        execution_result = await mcp_bridge.execute_function_call(func_call, user_id)

                        if execution_result.success:
                            logger.info(f"✅ Function call {func_call.name} executed successfully")

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
                                for result_part in follow_up_result.candidates[0].content.parts:
                                    if result_part.text:
                                        total_chunks += 1

                                        # Check if follow-up contains thinking
                                        is_thinking = hasattr(result_part, 'thought') and result_part.thought is True

                                        if is_thinking:
                                            thoughts += result_part.text
                                            thinking_chunks += 1
                                            has_thinking = True
                                        else:
                                            answer += result_part.text
                                            answer_chunks += 1

                        else:
                            # Function call failed - add error to answer
                            error_msg = f"\n[Function call {func_call.name} failed: {execution_result.error}]"
                            answer += error_msg
                            logger.error(f"❌ Function call {func_call.name} failed: {execution_result.error}")

                    except Exception as func_error:
                        error_msg = f"\n[Function call {func_call.name} error: {str(func_error)}]"
                        answer += error_msg
                        logger.error(f"❌ Function call {func_call.name} error: {func_error}")

            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            thinking_summary = self._create_thinking_summary(thoughts)

            # Include thinking in response if requested
            final_answer = answer
            if include_thinking_in_response and has_thinking and thoughts.strip():
                final_answer = f"**My Reasoning:**\n{thinking_summary}\n\n**My Response:**\n{answer}"

            logger.info(f"✅ Thinking+function processing complete for user {user_id}")
            logger.info(f"   📊 Total chunks: {total_chunks}")
            logger.info(f"   🧠 Thinking chunks: {thinking_chunks}")
            logger.info(f"   💬 Answer chunks: {answer_chunks}")
            logger.info(f"   🔧 Function calls: {function_calls_processed}")
            logger.info(f"   ⏱️ Processing time: {processing_time:.1f}ms")
            logger.info(f"   🎯 Has thinking: {has_thinking}")

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

    def _create_thinking_summary(self, thoughts: str, max_length: int = 200) -> str:
        """
        Create a concise summary of the AI's thinking process.

        Args:
            thoughts: Raw thinking text
            max_length: Maximum length for summary

        Returns:
            Summarized thinking process
        """
        # Ensure thoughts is actually a string
        if not isinstance(thoughts, str):
            thoughts = str(thoughts) if thoughts else ""

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
