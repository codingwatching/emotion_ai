### **Guide: Enabling AI Vision through Local Image File Processing**

**Objective:** To enable an AI to process local image files (e.g., `scene_01.png`) by reading their content, encoding them, and including them in multimodal prompts sent to a Gemini vision-capable model via its API.

**Core Concept:** Gemini's vision models (`gemini-1.5-flash-preview-0520` or similar multimodal models) can accept a list of "parts" in a single prompt, where each part can be text or image data. Local image files must be read as binary data, then Base64 encoded, and finally formatted as an `ImagePart` object for the Gemini API.

---

**Step-by-Step Implementation Guide:**

**1. Essential Imports:**
Ensure the following Python libraries are imported at the top of your main processing file (e.g., `aura_backend/main.py`). These are crucial for file path manipulation, regular expressions, and Base64 encoding.

```python
import base64
import re
from pathlib import Path # For robust path handling
from google.genai import types # For creating Gemini content parts
from typing import Optional # For type hinting in the helper function
```

**2. Create a Helper Function to Process Image Paths:**
This asynchronous helper function will take an image file path, read its binary content, Base64 encode it, and return a `types.Part` object suitable for Gemini. It should be placed outside of any class, or as a static method if part of a utility class.

```python
async def _get_image_part_from_path(image_path: str) -> Optional[types.Part]:
    """
    Reads an image file, encodes it to base64, and returns a types.Part for Gemini.
    This function uses a tool to execute Python code to read the binary data.
    """
    try:
        # Construct a Python script to read the file and base64 encode it
        read_and_encode_script = f"""
import base64
import os
from pathlib import Path

file_path = \"{image_path}\"
if os.path.exists(file_path) and Path(file_path).is_file():
    with open(file_path, \"rb\") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    print(encoded_image)
else:
    print(\"File not found\")
"""
        # Execute the script using an MCP tool (e.g., mcp_code_executor_execute_code)
        # Assumes 'execute_mcp_tool' is available and correctly configured
        result = await execute_mcp_tool(
            tool_name="mcp_code_executor_execute_code",
            arguments={"code": read_and_encode_script},
            user_id="aura_system" # Use a system user ID for internal tool calls
        )

        # Parse the result from the tool execution
        if result and result.get("result", {}).get("stdout"):
            encoded_image_str = result["result"]["stdout"].strip()

            if encoded_image_str and encoded_image_str != "File not found":
                # Determine MIME type from file extension
                ext = Path(image_path).suffix.lower()
                if ext == ".png":
                    mime_type = "image/png"
                elif ext in [".jpg", ".jpeg"]:
                    mime_type = "image/jpeg"
                elif ext == ".gif":
                    mime_type = "image/gif"
                # Add other image types as needed
                else:
                    logger.warning(f"Unsupported image format for vision: {ext}. Defaulting to image/jpeg.")
                    mime_type = "image/jpeg"

                return types.Part.from_data(data=encoded_image_str, mime_type=mime_type)
            else:
                logger.error(f"Image file not found or empty after reading: {image_path}")
        else:
            logger.error(f"Failed to get valid output from mcp_code_executor for image: {image_path}. Result: {result}")
        return None
    except Exception as e:
        logger.error(f"Error processing image path {image_path}: {e}")
        return None
```

**3. Modify the Main Conversation Processing Function (`process_conversation`):**
This is where the core logic for detecting image paths and constructing the multimodal prompt will reside.

```python
@app.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """
    Process conversation with MCP function calling, persistent chat context, and now, vision capabilities.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # ... (rest of the existing code for user profile, memory context, tool info) ...

        # --- NEW: Prepare content for Gemini (text and optional image) ---
        contents = []
        text_message_for_llm = request.message # Initialize with full message

        # Regex to detect a file path ending with common image extensions
        # This regex looks for a string that ends with a filename and image extension
        # It's flexible for various path formats (e.g., /path/to/image.png or just image.jpg)
        image_path_match = re.search(r'(?P<image_path>[\\/]?([a-zA-Z0-9_.-]+[\\/])*[a-zA-Z0-9_.-]+\\.(?:png|jpg|jpeg|gif|bmp))$', request.message, re.IGNORECASE)

        if image_path_match:
            detected_image_path = image_path_match.group('image_path')

            # Verify if the detected path actually exists as a file
            if Path(detected_image_path).is_file():
                image_part = await _get_image_part_from_path(detected_image_path)
                if image_part:
                    contents.append(image_part)
                    # Extract any accompanying text from the message
                    text_message_for_llm = request.message.replace(detected_image_path, '').strip()
                    if text_message_for_llm:
                        contents.append(types.Part.from_text(text_message_for_llm))
                    logger.info(f"🖼️ Image detected and added to Gemini content: {detected_image_path}")
                else:
                    logger.warning(f"Could not process image at {detected_image_path}. Sending message as text only.")
                    contents.append(types.Part.from_text(request.message))
            else:
                logger.debug(f"Detected string {detected_image_path} looks like an image path but is not a valid file. Sending as text.")
                contents.append(types.Part.from_text(request.message))
        else:
            # No image path detected, send as a regular text message
            contents.append(types.Part.from_text(request.message))

        # --- IMPORTANT: Update the Gemini model to a vision-capable one ---
        # Ensure your environment variable AURA_MODEL is set to a multimodal model
        # e.g., 'gemini-1.5-flash-002' or 'gemma-3-27b-it'
        # Fallback will be to a text-only model if env var is not set to a vision model.
        gemini_model_to_use = os.getenv('AURA_MODEL', 'gemini-1.5-flash-002')

        # --- Update chat session creation to use the new model and contents ---
        if needs_new_session:
            # ... (existing code to get gemini_tools) ...
            chat = client.chats.create(
                model=gemini_model_to_use, # Use the vision-capable model
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=int(os.getenv('AURA_MAX_OUTPUT_TOKENS', '1000000')),
                    tools=tools if tools else None,
                    system_instruction=system_instruction
                )
            )
            active_chat_sessions[session_key] = chat
            session_tool_versions[session_key] = global_tool_version
            logger.info(f"💬 Created new chat session for {request.user_id} with {len(tools)} tools (v{global_tool_version}) using model {gemini_model_to_use}")
        else:
            chat = active_chat_sessions[session_key]
            # Ensure the model in the existing chat is still vision-capable if needed,
            # though it should be handled by session recreation logic if global_tool_version changes.
            logger.debug(f"💬 Using existing chat session for {request.user_id}")


        # Send message and handle function calls
        # Pass the 'contents' list instead of just 'request.message'
        result = chat.send_message(contents) # <-- THIS IS THE CRUCIAL CHANGE

        # ... (rest of the existing code for processing function calls and generating final_response) ...

        aura_response = final_response or "I'm here and ready to help!"

        # --- IMPORTANT: Adjust Emotional/Cognitive Detection ---
        # For emotional/cognitive detection, use only the text part of the message
        # to avoid errors if the detection models are not multimodal.
        user_emotional_state = await detect_user_emotion(
            user_message=text_message_for_llm, # Use text_message_for_llm
            user_id=request.user_id
        )

        emotional_state_data = await detect_aura_emotion(
            conversation_snippet=f"User: {text_message_for_llm}\nAura: {aura_response}", # Use text_message_for_llm
            user_id=request.user_id
        )

        cognitive_state_data = await detect_aura_cognitive_focus(
            conversation_snippet=f"User: {text_message_for_llm}\nAura: {aura_response}", # Use text_message_for_llm
            user_id=request.user_id
        )

        # Create memory objects (store original message including path)
        user_memory = ConversationMemory(
            user_id=request.user_id,
            message=request.message, # Store original message including path for memory
            sender="user",
            emotional_state=user_emotional_state,
            session_id=session_id
        )

        # ... (rest of the existing code for storing memories, updating profile, and formatting response) ...

    except Exception as e:
        logger.error(f"❌ Failed to process conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
