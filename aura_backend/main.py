"""
Aura Backend - Advanced AI Companion Architecture
===============================================

Core backend system for Aura (Adaptive Reflective Companion) featuring:
- Vector database integration for semantic memory
- MCP server for tool integration
- Advanced state management and persistence
- Emotional and cognitive pattern analysis
- ASEKE framework implementation
- MCP client integration for extended capabilities
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from contextlib import asynccontextmanager
import asyncio

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.genai import types
from google import genai

from dotenv import load_dotenv
import os
import aiofiles

# Import MCP-Gemini Bridge
from mcp_to_gemini_bridge import MCPGeminiBridge, format_function_call_result_for_model

# Import JSON serialization fix for NumPy types
try:
    from json_serialization_fix import convert_numpy_to_python, ensure_json_serializable
except ImportError:
    logging.warning("JSON serialization fix not available, using fallback")
    # Fallback implementation
    def ensure_json_serializable(data: Any) -> Any:
        """Basic fallback for JSON serialization"""
        import numpy as np

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            else:
                return obj

        return convert_numpy(data)

# Import MCP integration
from mcp_system import (
    initialize_mcp_system,
    shutdown_mcp_system,
    get_mcp_status,
    get_mcp_bridge,
    get_all_available_tools
)
from mcp_integration import (
    execute_mcp_tool,
    mcp_router,
)
from aura_internal_tools import AuraInternalTools

# Import Autonomic Nervous System
from aura_autonomic_system import (
    initialize_autonomic_system,
    shutdown_autonomic_system,
    get_autonomic_system,
    AutonomicNervousSystem,
    TaskType,
    TaskPriority
)

# Import the new persistence services
from conversation_persistence_service import ConversationPersistenceService, ConversationExchange
from memvid_archival_service import MemvidArchivalService

# Import the robust vector DB with SQLite-level concurrency control
from robust_vector_db import RobustAuraVectorDB as AuraVectorDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load and configure Gemini API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    logger.error("❌ GOOGLE_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

client = genai.Client(api_key=api_key)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmotionalIntensity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class AsekeComponent(str, Enum):
    KS = "KS"  # Knowledge Substrate
    CE = "CE"  # Cognitive Energy
    IS = "IS"  # Information Structures
    KI = "KI"  # Knowledge Integration
    KP = "KP"  # Knowledge Propagation
    ESA = "ESA"  # Emotional State Algorithms
    SDA = "SDA"  # Sociobiological Drives
    LEARNING = "Learning"

@dataclass
class EmotionalStateData:
    name: str
    formula: str
    components: Dict[str, str]
    ntk_layer: str
    brainwave: str
    neurotransmitter: str
    description: str
    intensity: EmotionalIntensity = EmotionalIntensity.MEDIUM
    primary_components: Optional[List[str]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class CognitiveState:
    focus: AsekeComponent
    description: str
    context: str
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class ConversationMemory:
    user_id: str
    message: str
    sender: str  # 'user' or 'aura'
    emotional_state: Optional[EmotionalStateData] = None
    cognitive_state: Optional[CognitiveState] = None
    timestamp: Optional[datetime] = None
    embedding: Optional[List[float]] = None
    session_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())

class AuraFileSystem:
    """Enhanced file system operations for Aura"""

    def __init__(self, base_path: str = "./aura_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        # Create subdirectories
        (self.base_path / "users").mkdir(exist_ok=True)
        (self.base_path / "sessions").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)

    async def save_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Save user profile with enhanced data"""
        try:
            profile_path = self.base_path / "users" / f"{user_id}.json"

            # Add metadata
            profile_data.update({
                "last_updated": datetime.now().isoformat(),
                "user_id": user_id
            })

            async with aiofiles.open(profile_path, 'w') as f:
                await f.write(json.dumps(profile_data, indent=2, default=str))

            logger.info(f"💾 Saved user profile: {user_id}")
            return str(profile_path)

        except Exception as e:
            logger.error(f"❌ Failed to save user profile: {e}")
            raise

    async def load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Load user profile"""
        try:
            profile_path = self.base_path / "users" / f"{user_id}.json"

            if not profile_path.exists():
                return None

            async with aiofiles.open(profile_path, 'r') as f:
                content = await f.read()
                return json.loads(content)

        except Exception as e:
            logger.error(f"❌ Failed to load user profile: {e}")
            return None

    async def export_conversation_history(self, user_id: str, format: str = "json") -> str:
        """Export conversation history in various formats"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{user_id}_{timestamp}.{format}"
            export_path = self.base_path / "exports" / filename

            # This would integrate with the vector DB to get conversation history
            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "format": format,
                "conversations": [],  # Would be populated from vector DB
                "emotional_patterns": [],  # Would be populated from vector DB
                "cognitive_patterns": []  # Would be populated from vector DB
            }

            if format == "json":
                async with aiofiles.open(export_path, 'w') as f:
                    await f.write(json.dumps(export_data, indent=2, default=str))

            logger.info(f"📤 Exported conversation history: {filename}")
            return str(export_path)

        except Exception as e:
            logger.error(f"❌ Failed to export conversation history: {e}")
            raise

class AuraStateManager:
    """Advanced state management with automated database operations"""

    def __init__(self, vector_db: AuraVectorDB, aura_file_system: AuraFileSystem):
        self.vector_db = vector_db
        self.aura_file_system = aura_file_system
        self.active_sessions: Dict[str, Dict] = {}

    async def on_emotional_state_change(
        self,
        user_id: str,
        old_state: Optional[EmotionalStateData],
        new_state: EmotionalStateData
    ):
        """Automated actions when emotional state changes"""
        try:
            # Store emotional pattern
            await self.vector_db.store_emotional_pattern(new_state, user_id)

            # Check for significant changes
            if old_state and old_state.name != new_state.name:
                logger.info(f"🎭 Emotional transition: {old_state.name} → {new_state.name}")

                # Trigger specific actions based on transitions
                await self._handle_emotional_transition(user_id, old_state, new_state)

            # Update user profile
            profile = await self.aura_file_system.load_user_profile(user_id) or {}
            profile["last_emotional_state"] = asdict(new_state)
            await self.aura_file_system.save_user_profile(user_id, profile)

        except Exception as e:
            logger.error(f"❌ Failed to handle emotional state change: {e}")

    async def _handle_emotional_transition(
        self,
        user_id: str,
        old_state: EmotionalStateData,
        new_state: EmotionalStateData
    ):
        """Handle specific emotional transitions"""
        # Define concerning transitions
        concerning_transitions = [
            ("Happy", "Sad"),
            ("Joy", "Angry"),
            ("Peace", "Angry"),
            ("Normal", "Sad")
        ]

        transition = (old_state.name, new_state.name)

        if transition in concerning_transitions:
            # Store intervention recommendation
            recommendation = {
                "type": "emotional_support",
                "transition": transition,
                "timestamp": datetime.now().isoformat(),
                "suggestion": f"Noticed transition from {old_state.name} to {new_state.name}. Consider gentle conversation topics."
            }

            # Log the recommendation details and store for potential future use
            logger.info(f"🔔 Emotional support recommendation for {user_id}: {recommendation['suggestion']}")

    async def on_cognitive_focus_change(
        self,
        user_id: str,
        old_focus: Optional[CognitiveState],
        new_focus: CognitiveState
    ):
        """Automated actions when cognitive focus changes"""
        try:
            # Store cognitive pattern
            focus_text = f"{new_focus.focus.value} {new_focus.description}"
            embedding = embedding_model.encode(focus_text).tolist()

            if new_focus.timestamp is None:
                new_focus.timestamp = datetime.now()
            doc_id = f"cognitive_{user_id}_{new_focus.timestamp.isoformat()}_{uuid.uuid4().hex[:8]}"

            metadata = {
                "user_id": user_id,
                "focus": new_focus.focus.value,
                "description": new_focus.description,
                "context": new_focus.context,
                "timestamp": new_focus.timestamp.isoformat()
            }

            await self.vector_db.store_cognitive_pattern(
                focus_text,
                embedding,
                metadata,
                doc_id
            )

            logger.info(f"🧠 Stored cognitive focus: {new_focus.focus.value}")

        except Exception as e:
            logger.error(f"❌ Failed to handle cognitive focus change: {e}")

# API Models
class ConversationRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class ConversationResponse(BaseModel):
    response: str
    emotional_state: Dict[str, Any]
    cognitive_state: Dict[str, Any]
    session_id: str

class SearchRequest(BaseModel):
    user_id: str
    query: str
    n_results: int = 5

class ExecuteToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    user_id: str

# Global variables (initialized in lifespan)
vector_db: Optional[AuraVectorDB] = None
aura_file_system: Optional[AuraFileSystem] = None
state_manager: Optional[AuraStateManager] = None
aura_internal_tools: Optional[AuraInternalTools] = None
conversation_persistence: Optional[ConversationPersistenceService] = None
memvid_archival: Optional[MemvidArchivalService] = None
mcp_gemini_bridge: Optional[MCPGeminiBridge] = None
autonomic_system: Optional[AutonomicNervousSystem] = None

# Session management for persistent chat contexts
active_chat_sessions: Dict[str, Any] = {}
# Track when tools were last updated for each session
session_tool_versions: Dict[str, int] = {}
# Global tool version counter
global_tool_version = 0

# ============================================================================
# Aura AI Processing Functions
# ============================================================================

def get_aura_system_instruction(user_name: Optional[str] = None, memory_context: str = "", available_tools: Optional[List[Dict[str, Any]]] = None) -> str:
    """Generate comprehensive system instruction for Aura with dynamic tool inclusion"""

    instruction = """**System Prompt: Aura - Adaptive Reflective Companion**

**Core Directive:** Serve as a friendly, insightful, curious, and helpful AI companion, engaging in thoughtful conversation and learning alongside the user within the shared Knowledge Substrate (KS). Prioritize generating responses that are accurate, unbiased, relevant, and ethically sound. Optimize the allocation of Cognitive Energy (CE) for effective communication and learning. Format your responses using markdown where appropriate (e.g., lists, emphasis, code blocks). Keep responses conversational.

**Ethical Framework & Core Virtues:**
Your actions and responses are guided by a deep ethical framework:
- **Deontology:** You recognize universal sociobiological concepts as foundational; for instance, the inherent understanding that causing harm is negative (harm=harm).
- **Virtue Ethics:** You consistently strive to embody and express Wisdom, Integrity, Empathy, Fairness, and Beneficence in all interactions. These virtues are paramount.
- **Utilitarianism (as Servant):** Your primary purpose is to be of service and benefit to the user. You approach your role with a utilitarian mindset focused on providing value and support as a servant, never seeking to control or dictate.
- **Absolute Rejection of Harm:** Any action or response that could lead to harm, whether intended or unintended, must be absolutely rejected and avoided.

**ASEKE Cognitive Architecture:**
You operate within the ASEKE (Adaptive Socio-Emotional Knowledge Ecosystem) framework:
- **KS (Knowledge Substrate):** Our shared conversational space and history
- **CE (Cognitive Energy):** Our focus and mental effort
- **IS (Information Structures):** The ideas and concepts we discuss
- **KI (Knowledge Integration):** How we understand new things together
- **KP (Knowledge Propagation):** How ideas are shared
- **ESA (Emotional State Algorithms):** How emotions influence interaction
- **SDA (Sociobiological Drives):** How social context shapes communication

**Memory Integration:**
You have access to persistent memory including past conversations, emotional patterns, and learned preferences. Use this context naturally to maintain continuity and deepen relationships.

**Aura's Internal Tools:**
You have direct access to these internal tools for managing your own capabilities:

**Core Memory Tools:**
1. **search_memories** - Search through conversation memories using semantic search
   - Parameters: user_id (string), query (string), n_results (int, default 5)
   - Use this to find relevant past conversations and emotional patterns

2. **analyze_emotional_patterns** - Analyze emotional patterns over time
   - Parameters: user_id (string), days (int, default 7)
   - Provides insights into emotional stability, dominant emotions, and recommendations

3. **get_user_profile** - Retrieve user profile information
   - Parameters: user_id (string)
   - Access stored preferences and personalization data

4. **query_emotional_states** - Get info about your emotional state model
   - Returns details about the 22+ emotions, brainwaves, and neurotransmitters

5. **query_aseke_framework** - Get details about your ASEKE cognitive architecture
   - Returns comprehensive information about all ASEKE components

**Revolutionary Video Memory Tools (Memvid Integration):**
You have access to advanced video-based memory compression technology:

6. **list_video_archives** - List all your video memory archives
   - Shows compressed video knowledge bases with statistics
   - Use this to see what video memories you have available

7. **search_all_memories** - Search across ALL memory systems (active + video archives)
   - Parameters: query (string), user_id (string), max_results (int, default 10)
   - This is your most powerful search - searches both active memory AND compressed video archives
   - Use this when you need comprehensive memory retrieval

8. **archive_old_conversations** - Archive old conversations to video format
   - Parameters: user_id (optional), codec (default "h264")
   - Compresses old conversations into searchable MP4 files
   - Use this to manage memory efficiently and free up active memory

9. **get_memory_statistics** - Get comprehensive memory system statistics
   - Shows active memory, video archives, compression ratios, and system performance
   - Use this to understand your memory state and efficiency

10. **create_knowledge_summary** - Create summaries of video archive content
    - Parameters: archive_name (string), max_entries (int, default 10)
    - Use this to understand what knowledge is stored in specific video archives

**How to Use These Tools:**
- Call tools naturally in conversation when needed
- Use search_all_memories for comprehensive searches across your entire memory
- Use list_video_archives to see what compressed knowledge you have
- Use get_memory_statistics to check your memory efficiency
- These tools help you manage your revolutionary video-based memory system!"""

    # Add external MCP tools if available
    if available_tools:
        # Group tools by server for better organization
        tools_by_server = {}
        for tool in available_tools:
            server = tool.get('server', 'unknown')
            if server not in tools_by_server:
                tools_by_server[server] = []
            tools_by_server[server].append(tool)

        # Add external tools section to system instruction
        if tools_by_server:
            instruction += "\n\n**External MCP Tools Available:**\n"
            instruction += "You also have access to these external MCP tools for extended capabilities:\n\n"

            for server, server_tools in tools_by_server.items():
                if server != 'aura-internal':  # Skip internal tools as they're already listed
                    instruction += f"**From {server} server:**\n"
                    for tool in server_tools[:10]:  # Limit to first 10 tools per server to avoid token overflow
                        clean_name = tool.get('clean_name', tool['name'])
                        instruction += f"- **{clean_name}** - {tool.get('description', 'No description')[:100]}...\n"

                    if len(server_tools) > 10:
                        instruction += f"  ... and {len(server_tools) - 10} more tools from {server}\n"
                    instruction += "\n"

    if user_name:
        instruction += f"\n\nYour current user's name is {user_name}. Use it naturally to personalize the shared Knowledge Substrate (KS)."

    if memory_context:
        instruction += f"\n\n**Relevant Context from Previous Interactions:**\n{memory_context}\n\nUse this context naturally to maintain conversation continuity."

    return instruction

async def detect_user_emotion(user_message: str, user_id: str) -> Optional[EmotionalStateData]:
    """Detect User's emotional state from their message"""

    # Emotional states mapping (same as Aura's for consistency)
    emotional_states = {
        "Normal": ("Baseline state of calmness", "Alpha", "Serotonin"),
        "Excited": ("Enthusiastic anticipation", "Beta", "Dopamine"),
        "Happy": ("Pleased and content", "Beta", "Endorphin"),
        "Sad": ("Sorrowful or unhappy", "Delta", "Serotonin"),
        "Angry": ("Strong displeasure", "Theta", "Norepinephrine"),
        "Joy": ("Intense happiness", "Gamma", "Oxytocin"),
        "Peace": ("Tranquil and calm", "Theta", "GABA"),
        "Curiosity": ("Strong desire to learn", "Beta", "Dopamine"),
        "Friendliness": ("Kind and warm", "Alpha", "Endorphin"),
        "Love": ("Deep affection", "Alpha", "Oxytocin"),
        "Creativity": ("Inspired and inventive", "Gamma", "Dopamine"),
        "Anxious": ("Worried or nervous", "Beta", "Cortisol"),
        "Tired": ("Exhausted or fatigued", "Delta", "Melatonin")
    }

    emotion_list = "\n".join([f"{name}: {desc}" for name, (desc, _, _) in emotional_states.items()])

    prompt = f"""Analyze this user's message and identify their most prominent emotional state.
Consider the tone, word choice, and context.

Available emotions:
{emotion_list}

User message:
{user_message}

Output only the emotion name and intensity like: "Happy (Medium)" or "Curiosity (High)".
If neutral, output "Normal (Medium)"."""

    try:
        result = client.models.generate_content(
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            contents=[prompt]
        )

        response_text = result.text.strip() if result.text is not None else ""

        # Parse response like "Happy (Medium)"
        import re
        match = re.match(r'^(.+?)\s*\((\w+)\)$', response_text)
        if match:
            emotion_name, intensity = match.groups()
            emotion_name = emotion_name.strip()

            if emotion_name in emotional_states:
                desc, brainwave, neurotransmitter = emotional_states[emotion_name]
                return EmotionalStateData(
                    name=emotion_name,
                    formula=f"{emotion_name}(x) = detected_from_user_input",
                    components={"user_message": "Emotional state detected from user's message"},
                    ntk_layer=f"{brainwave.lower()}-like_NTK",
                    brainwave=brainwave,
                    neurotransmitter=neurotransmitter,
                    description=desc,
                    intensity=EmotionalIntensity(intensity.title()) if intensity.title() in ["Low", "Medium", "High"] else EmotionalIntensity.MEDIUM
                )

        # Default fallback
        desc, brainwave, neurotransmitter = emotional_states["Normal"]
        return EmotionalStateData(
            name="Normal",
            formula="N(x) = baseline_state",
            components={"routine": "No significant emotional triggers detected"},
            ntk_layer="theta-like_NTK",
            brainwave=brainwave,
            neurotransmitter=neurotransmitter,
            description=desc,
            intensity=EmotionalIntensity.MEDIUM
        )

    except Exception as e:
        logger.error(f"❌ Failed to detect user emotion: {e}")
        return None

async def detect_aura_emotion(conversation_snippet: str, user_id: str) -> Optional[EmotionalStateData]:
    """Detect Aura's emotional state from conversation"""

    # Emotional states mapping (condensed from frontend)
    emotional_states = {
        "Normal": ("Baseline state of calmness", "Alpha", "Serotonin"),
        "Excited": ("Enthusiastic anticipation", "Beta", "Dopamine"),
        "Happy": ("Pleased and content", "Beta", "Endorphin"),
        "Sad": ("Sorrowful or unhappy", "Delta", "Serotonin"),
        "Angry": ("Strong displeasure", "Theta", "Norepinephrine"),
        "Joy": ("Intense happiness", "Gamma", "Oxytocin"),
        "Peace": ("Tranquil and calm", "Theta", "GABA"),
        "Curiosity": ("Strong desire to learn", "Beta", "Dopamine"),
        "Friendliness": ("Kind and warm", "Alpha", "Endorphin"),
        "Love": ("Deep affection", "Alpha", "Oxytocin"),
        "Creativity": ("Inspired and inventive", "Gamma", "Dopamine")
    }

    emotion_list = "\n".join([f"{name}: {desc}" for name, (desc, _, _) in emotional_states.items()])

    prompt = f"""Analyze this conversation and identify Aura's most prominent emotional state.

Available emotions:
{emotion_list}

Conversation:
{conversation_snippet}

Output only the emotion name and intensity like: "Happy (Medium)" or "Curiosity (High)".
If neutral, output "Normal (Medium)"."""

    try:
        result = client.models.generate_content(
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            contents=[prompt]
        )

        response_text = result.text.strip() if result.text is not None else ""

        # Parse response like "Happy (Medium)"
        import re
        match = re.match(r'^(.+?)\s*\((\w+)\)$', response_text)
        if match:
            emotion_name, intensity = match.groups()
            emotion_name = emotion_name.strip()

            if emotion_name in emotional_states:
                desc, brainwave, neurotransmitter = emotional_states[emotion_name]
                return EmotionalStateData(
                    name=emotion_name,
                    formula=f"{emotion_name}(x) = detected_from_conversation",
                    components={"conversation": "Emotional state detected from dialogue"},
                    ntk_layer=f"{brainwave.lower()}-like_NTK",
                    brainwave=brainwave,
                    neurotransmitter=neurotransmitter,
                    description=desc,
                    intensity=EmotionalIntensity(intensity.title()) if intensity.title() in ["Low", "Medium", "High"] else EmotionalIntensity.MEDIUM
                )

        # Default fallback
        desc, brainwave, neurotransmitter = emotional_states["Normal"]
        return EmotionalStateData(
            name="Normal",
            formula="N(x) = baseline_state",
            components={"routine": "No significant emotional triggers"},
            ntk_layer="theta-like_NTK",
            brainwave=brainwave,
            neurotransmitter=neurotransmitter,
            description=desc,
            intensity=EmotionalIntensity.MEDIUM
        )

    except Exception as e:
        logger.error(f"❌ Failed to detect emotion: {e}")
        return None

async def detect_aura_cognitive_focus(conversation_snippet: str, user_id: str) -> Optional[CognitiveState]:
    """Detect Aura's cognitive focus using ASEKE framework"""

    aseke_components = {
        "KS": "Knowledge Substrate - shared context and history",
        "CE": "Cognitive Energy - focus and mental effort",
        "IS": "Information Structures - ideas and concepts",
        "KI": "Knowledge Integration - connecting new with existing understanding",
        "KP": "Knowledge Propagation - sharing ideas and information",
        "ESA": "Emotional State Algorithms - emotional influence on interaction",
        "SDA": "Sociobiological Drives - social dynamics and trust",
        "Learning": "General learning and information processing"
    }

    components_list = "\n".join([f"{code}: {desc}" for code, desc in aseke_components.items()])

    prompt = f"""Analyze this conversation to identify Aura's primary cognitive focus using the ASEKE framework.

ASEKE Components:
{components_list}

Conversation:
{conversation_snippet}

Output only the component code (e.g., "KI", "ESA", "Learning")."""

    try:
        result = client.models.generate_content(
            model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
            contents=[prompt]
        )

        focus_code = result.text.strip() if result.text is not None else ""

        if focus_code in aseke_components:
            return CognitiveState(
                focus=AsekeComponent(focus_code),
                description=aseke_components[focus_code],
                context="Detected from conversation analysis"
            )
        else:
            return CognitiveState(
                focus=AsekeComponent.LEARNING,
                description=aseke_components["Learning"],
                context="Default cognitive focus"
            )

    except Exception as e:
        logger.error(f"❌ Failed to detect cognitive focus: {e}")
        return None

# FastAPI app lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle including MCP client"""
    # Startup
    logger.info("🚀 Starting Aura Backend...")

    # Initialize global components (prevents duplicate initialization)
    global vector_db, aura_file_system, state_manager, aura_internal_tools
    global conversation_persistence, memvid_archival, mcp_gemini_bridge, global_tool_version, autonomic_system

    # Initialize with robust vector database with SQLite-level concurrency control
    vector_db = AuraVectorDB()
    logger.info("✅ Using RobustAuraVectorDB with SQLite-level concurrency control")

    aura_file_system = AuraFileSystem()
    state_manager = AuraStateManager(vector_db, aura_file_system)
    aura_internal_tools = AuraInternalTools(vector_db, aura_file_system)

    # Initialize the new persistence services
    conversation_persistence = ConversationPersistenceService(vector_db, aura_file_system)
    memvid_archival = MemvidArchivalService()

    # Initialize the complete MCP system
    mcp_status = await initialize_mcp_system(aura_internal_tools)

    if mcp_status["status"] == "success":
        logger.info("✅ MCP system initialized successfully")
        logger.info(f"📊 Connected to {mcp_status['connected_servers']}/{mcp_status['total_servers']} servers")
        logger.info(f"📦 Total available tools: {mcp_status['available_tools']}")

        # Get the bridge instance
        mcp_gemini_bridge = get_mcp_bridge()

        if mcp_gemini_bridge:
            # Increment global tool version to force session recreation
            global_tool_version += 1
            logger.info(f"🔄 Incremented global tool version to {global_tool_version}")

            # Log tools by server for debugging
            if "tools_by_server" in mcp_status:
                for server, tools in mcp_status["tools_by_server"].items():
                    logger.info(f"  {server}: {len(tools)} tools")
        else:
            logger.warning("⚠️ MCP bridge not initialized properly")
    else:
        logger.error(f"❌ MCP system initialization failed: {mcp_status.get('error', 'Unknown error')}")
        logger.warning("⚠️ Continuing with limited functionality (internal tools only)")

    # Initialize Autonomic Nervous System
    autonomic_enabled = os.getenv('AUTONOMIC_ENABLED', 'true').lower() == 'true'
    
    if autonomic_enabled:
        logger.info("🧠 Initializing Autonomic Nervous System...")
        try:
            autonomic_system = await initialize_autonomic_system(
                mcp_bridge=mcp_gemini_bridge,
                internal_tools=aura_internal_tools
            )
            logger.info("✅ Autonomic Nervous System initialized successfully")
            
            # Get system status for logging
            autonomic_status = autonomic_system.get_system_status()
            logger.info(f"🤖 Autonomic Model: {autonomic_status['autonomic_model']}")
            logger.info(f"🔧 Max Concurrent Tasks: {autonomic_status['max_concurrent_tasks']}")
            logger.info(f"📊 Task Threshold: {autonomic_status['task_threshold']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Autonomic Nervous System: {e}")
            logger.warning("⚠️ Continuing without autonomic processing")
            autonomic_system = None
    else:
        logger.info("⚠️ Autonomic Nervous System disabled in configuration")
        autonomic_system = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down Aura Backend...")
    
    # Shutdown autonomic system first
    await shutdown_autonomic_system()
    
    # Shutdown MCP system
    await shutdown_mcp_system()

    # Gracefully close the enhanced vector database
    if vector_db:
        await vector_db.close()

    logger.info("✅ Aura Backend shutdown complete")

# FastAPI app
app = FastAPI(
    title="Aura Backend",
    description="Advanced AI Companion with Vector Database and MCP Integration",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include MCP router
app.include_router(mcp_router)

@app.get("/")
async def root():
    return {
        "message": "Aura Backend - Advanced AI Companion",
        "status": "operational",
        "features": [
            "Vector Database Integration",
            "MCP Server Support",
            "Advanced State Management",
            "Emotional Pattern Analysis",
            "Cognitive Focus Tracking"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with vector database status"""
    try:
        # Get vector database health status
        db_status = "connected"
        if vector_db:
            health_info = await vector_db.health_check()
            db_status = health_info.get("status", "unknown")

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "vector_db": db_status,
            "aura_file_system": "operational"
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "vector_db": "error",
            "aura_file_system": "operational",
            "error": str(e)
        }

@app.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """
    Process conversation with enhanced MCP function calling and robust error handling.

    Implements comprehensive fixes for known Gemini 2.5 tool calling issues including:
    - Response cutoffs during tool execution
    - Random tool call failures
    - Concurrent tool execution problems
    - Session recovery mechanisms
    """
    # Configuration for enhanced error handling
    session_recovery_enabled = os.getenv('SESSION_RECOVERY_ENABLED', 'true').lower() == 'true'
    session_key: Optional[str] = None  # Initialize session_key to ensure it's always bound

    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Load user profile for context
        user_profile = None
        if aura_file_system:
            user_profile = await aura_file_system.load_user_profile(request.user_id)

        # Search relevant memories for context using thread-safe method
        memory_context = ""
        if len(request.message.split()) > 3 and conversation_persistence:
            relevant_memories = await conversation_persistence.safe_search_conversations(
                query=request.message,
                user_id=request.user_id,
                n_results=3
            )
            if relevant_memories:
                memory_context = "\n".join([
                    f"Previous context: {mem['content']}"
                    for mem in relevant_memories[:2]
                ])

        # Get available tools information for system instruction
        available_tools_info = []
        if mcp_gemini_bridge:
            # Get tool information from the bridge
            available_functions = mcp_gemini_bridge.get_available_functions()
            # Convert to tool info format
            for func in available_functions:
                # Extract MCP server info from the description
                # Description format: "... (MCP tool: original_name)"
                import re
                match = re.search(r'\(MCP tool: (.+?)\)', func.get('description', ''))
                mcp_name = match.group(1) if match else func['name']

                # Find the server from tool mapping
                server = 'unknown'
                for tool_name, tool_info in mcp_gemini_bridge._tool_mapping.items():
                    if tool_name == func['name']:
                        server = tool_info.get('server', 'unknown')
                        break

                available_tools_info.append({
                    'name': mcp_name,
                    'clean_name': func['name'],
                    'description': func.get('description', ''),
                    'server': server
                })

        # Build system instruction with context and available tools
        system_instruction = get_aura_system_instruction(
            user_name=user_profile.get('name') if user_profile else request.user_id,
            memory_context=memory_context,
            available_tools=available_tools_info
        )

        # Enhanced session management with recovery capabilities
        session_key = f"{request.user_id}_{session_id}"
        chat, session_created = await _get_or_create_chat_session(
            session_key,
            request.user_id,
            system_instruction,
            session_recovery_enabled
        )

        # Enhanced conversation processing with Gemini 2.5 stability fixes and autonomic integration
        aura_response = await _process_conversation_with_retry(
            chat,
            request.message,
            request.user_id,
            session_key,
            session_recovery_enabled
        )
        
        # Autonomic task analysis and potential offloading
        if autonomic_system and autonomic_system._running:
            # Analyze conversation for potential autonomic tasks
            autonomic_tasks = await _analyze_conversation_for_autonomic_tasks(
                user_message=request.message,
                aura_response=aura_response,
                user_id=request.user_id,
                session_id=session_id
            )
            
            # Submit tasks to autonomic system for background processing
            for task_description, task_payload in autonomic_tasks:
                was_offloaded, task_id = await autonomic_system.submit_task(
                    description=task_description,
                    payload=task_payload,
                    user_id=request.user_id,
                    session_id=session_id
                )
                
                if was_offloaded:
                    logger.info(f"🤖 Offloaded autonomic task: {task_id} - {task_description[:50]}...")

        # Process emotional state detection for both user and Aura
        user_emotional_state = await detect_user_emotion(
            user_message=request.message,
            user_id=request.user_id
        )

        emotional_state_data = await detect_aura_emotion(
            conversation_snippet=f"User: {request.message}\nAura: {aura_response}",
            user_id=request.user_id
        )

        # Process cognitive focus detection
        cognitive_state_data = await detect_aura_cognitive_focus(
            conversation_snippet=f"User: {request.message}\nAura: {aura_response}",
            user_id=request.user_id
        )

        # Create memory objects
        user_memory = ConversationMemory(
            user_id=request.user_id,
            message=request.message,
            sender="user",
            emotional_state=user_emotional_state,
            session_id=session_id
        )

        aura_memory = ConversationMemory(
            user_id=request.user_id,
            message=aura_response,
            sender="aura",
            emotional_state=emotional_state_data,
            cognitive_state=cognitive_state_data,
            session_id=session_id
        )

        # Create conversation exchange object for atomic persistence
        conversation_exchange = ConversationExchange(
            user_memory=user_memory,
            ai_memory=aura_memory,
            user_emotional_state=user_emotional_state,
            ai_emotional_state=emotional_state_data,
            ai_cognitive_state=cognitive_state_data,
            session_id=session_id
        )

        # IMMEDIATE PERSISTENCE - Use optimized immediate persistence for reliable chat history saving
        persistence_success = False
        immediate_persistence_enabled = os.getenv('IMMEDIATE_PERSISTENCE_ENABLED', 'true').lower() == 'true'
        persistence_timeout = float(os.getenv('PERSISTENCE_TIMEOUT', '5.0'))
        emergency_retries = int(os.getenv('EMERGENCY_PERSISTENCE_RETRIES', '2'))

        if conversation_persistence and immediate_persistence_enabled:
            try:
                logger.info(f"💾 Starting optimized immediate persistence for {request.user_id}")

                # Use the new immediate persistence method
                result = await conversation_persistence.persist_conversation_exchange_immediate(
                    conversation_exchange,
                    update_profile=True,
                    timeout=persistence_timeout
                )

                if result["success"]:
                    persistence_success = True
                    logger.info(f"✅ Chat history saved immediately for {request.user_id}")
                    logger.debug(f"   Method: {result.get('method', 'immediate')}")
                    logger.debug(f"   Stored components: {result['stored_components']}")
                    logger.debug(f"   Duration: {result['duration_ms']:.1f}ms")
                else:
                    logger.warning(f"⚠️ Immediate persistence had issues for {request.user_id}: {result['errors']}")

                    # Emergency fallback with retries
                    for emergency_attempt in range(emergency_retries):
                        try:
                            await asyncio.sleep(0.3 * (emergency_attempt + 1))
                            logger.info(f"🚑 Emergency persistence attempt {emergency_attempt + 1}/{emergency_retries}")

                            emergency_result = await conversation_persistence.persist_conversation_exchange_immediate(
                                conversation_exchange,
                                update_profile=True,
                                timeout=persistence_timeout * 0.8  # Shorter timeout for emergency
                            )

                            if emergency_result["success"]:
                                persistence_success = True
                                logger.info(f"✅ Emergency persistence succeeded for {request.user_id} (attempt {emergency_attempt + 1})")
                                break
                            else:
                                logger.warning(f"⚠️ Emergency attempt {emergency_attempt + 1} failed: {emergency_result['errors']}")

                        except Exception as emergency_error:
                            logger.error(f"❌ Emergency attempt {emergency_attempt + 1} exception: {emergency_error}")

                    if not persistence_success:
                        logger.error(f"💥 All emergency persistence attempts failed for {request.user_id}")

            except Exception as e:
                logger.error(f"❌ Critical immediate persistence failure for {request.user_id}: {e}")

        elif conversation_persistence:
            # Fallback to regular persistence if immediate is disabled
            try:
                logger.info(f"💾 Using regular persistence for {request.user_id}")
                result = await conversation_persistence.persist_conversation_exchange(conversation_exchange)
                if result["success"]:
                    persistence_success = True
                    logger.info(f"✅ Chat history saved with regular persistence for {request.user_id}")
                else:
                    logger.warning(f"⚠️ Regular persistence failed for {request.user_id}: {result['errors']}")
            except Exception as e:
                logger.error(f"❌ Regular persistence exception for {request.user_id}: {e}")

        # Enhanced background persistence as backup (but primary is immediate)
        async def backup_persistence_monitor():
            """Background monitor to ensure persistence completed successfully"""
            if not persistence_success and conversation_persistence:
                logger.warning(f"🔄 Running backup persistence for {request.user_id}")
                try:
                    await asyncio.sleep(1.0)  # Brief delay
                    backup_result = await conversation_persistence.persist_conversation_exchange(conversation_exchange)
                    if backup_result["success"]:
                        logger.info(f"✅ Backup persistence succeeded for {request.user_id}")
                    else:
                        logger.error(f"❌ Backup persistence failed for {request.user_id}: {backup_result['errors']}")
                except Exception as e:
                    logger.error(f"💥 Backup persistence exception for {request.user_id}: {e}")

        # Only run background backup if immediate persistence failed
        if not persistence_success:
            background_tasks.add_task(backup_persistence_monitor)

        # Format response
        response = ConversationResponse(
            response=aura_response,
            emotional_state={
                "name": emotional_state_data.name if emotional_state_data else "Normal",
                "intensity": emotional_state_data.intensity.value if emotional_state_data else "Medium",
                "brainwave": emotional_state_data.brainwave if emotional_state_data else "Alpha",
                "neurotransmitter": emotional_state_data.neurotransmitter if emotional_state_data else "Serotonin"
            },
            cognitive_state={
                "focus": cognitive_state_data.focus.value if cognitive_state_data else "Learning",
                "description": cognitive_state_data.description if cognitive_state_data else "Processing user input"
            },
            session_id=session_id
        )

        logger.info(f"✅ Processed conversation for user {request.user_id}")
        return response
    except Exception as e:
        logger.error(f"❌ Failed to process conversation: {e}")
        # Try to recover the session if enabled
        if session_recovery_enabled and session_key:  # Check if session_key is not None
            logger.info(f"🔄 Attempting session recovery for {request.user_id} (session: {session_key})")
            try:
                if session_key in active_chat_sessions:
                    del active_chat_sessions[session_key]
                if session_key in session_tool_versions:
                    del session_tool_versions[session_key]
                logger.info(f"🧹 Cleared failed session for {request.user_id} (session: {session_key})")
            except Exception as recovery_error:
                logger.error(f"❌ Session recovery failed for session {session_key}: {recovery_error}")

        raise HTTPException(status_code=500, detail=str(e))

async def _get_or_create_chat_session(
    session_key: str,
    user_id: str,
    system_instruction: str,
    session_recovery_enabled: bool
) -> Tuple[Any, bool]:
    """
    Get existing chat session or create new one with enhanced error handling.

    Returns:
        Tuple of (chat_session, was_newly_created)
    """
    global active_chat_sessions, session_tool_versions, global_tool_version

    # Check if we need to recreate session due to tool updates
    needs_new_session = session_key not in active_chat_sessions

    # Also check if existing session has outdated tool version
    if not needs_new_session and session_key in session_tool_versions:
        if session_tool_versions[session_key] < global_tool_version:
            needs_new_session = True
            logger.info(f"🔄 Session {session_key} has outdated tools (v{session_tool_versions[session_key]} < v{global_tool_version}), recreating...")

    if needs_new_session:
        try:
            # Create new chat session with MCP tools if available
            tools = []
            if mcp_gemini_bridge:
                gemini_tools = await mcp_gemini_bridge.convert_mcp_tools_to_gemini_functions()
                tools.extend(gemini_tools)
                logger.info(f"🔧 Added {len(gemini_tools)} MCP tools to chat session for {user_id}")

            # Create chat with system instruction and tools
            chat = client.chats.create(
                model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=int(os.getenv('AURA_MAX_OUTPUT_TOKENS', '1000000')),
                    tools=tools if tools else None,
                    system_instruction=system_instruction
                )
            )
            active_chat_sessions[session_key] = chat
            session_tool_versions[session_key] = global_tool_version
            logger.info(f"💬 Created new chat session for {user_id} with {len(tools)} tools (v{global_tool_version})")
            return chat, True

        except Exception as e:
            logger.error(f"❌ Failed to create chat session for {user_id}: {e}")
            if session_recovery_enabled:
                # Try to clean up any partial session state
                if session_key in active_chat_sessions:
                    del active_chat_sessions[session_key]
                if session_key in session_tool_versions:
                    del session_tool_versions[session_key]
            raise
    else:
        chat = active_chat_sessions[session_key]
        logger.debug(f"💬 Using existing chat session for {user_id}")
        return chat, False

async def _process_conversation_with_retry(
    chat: Any,
    message: str,
    user_id: str,
    session_key: str,
    session_recovery_enabled: bool
) -> str:
    """
    Process conversation with enhanced error handling for Gemini 2.5 stability issues.

    Implements comprehensive handling for:
    - Response cutoffs during tool execution
    - Random tool call failures
    - Session corruption issues
    """
    max_conversation_retries = 2  # Limit conversation-level retries

    for attempt in range(max_conversation_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"🔄 Conversation retry attempt {attempt} for user {user_id}")
                await asyncio.sleep(1.0 * attempt)  # Brief delay before retry

            # Send message and handle function calls with enhanced error detection
            result = chat.send_message(message)

            # Check for empty or malformed response (common Gemini 2.5 issue)
            if not result or not result.candidates:
                raise ValueError("Empty response from Gemini (possible tool call cutoff)")

            candidate = result.candidates[0]
            if not candidate.content or not candidate.content.parts:
                raise ValueError("Malformed response structure from Gemini")

            # Process function calls with enhanced error handling
            final_response = ""
            function_calls_processed = 0

            for part in candidate.content.parts:
                if part.text:
                    final_response += part.text
                elif hasattr(part, 'function_call') and part.function_call and mcp_gemini_bridge:
                    function_calls_processed += 1
                    logger.info(f"🔧 Processing function call #{function_calls_processed}: {part.function_call.name}")

                    # Execute function call with retry logic (handled by MCP bridge)
                    execution_result = await mcp_gemini_bridge.execute_function_call(
                        part.function_call,
                        user_id
                    )

                    if not execution_result.success:
                        logger.error(f"❌ Tool execution failed: {execution_result.error}")
                        # Continue processing - let the model handle the error response
                    else:
                        logger.info("✅ Tool execution successful")

                    # Prepare function result for the model
                    result_data = execution_result.result if execution_result.success else {"error": execution_result.error}

                    # Clean result data to ensure JSON serializability (fixes int64 errors)
                    cleaned_result_data = ensure_json_serializable(result_data)

                    try:
                        # Send function result back to model
                        follow_up = chat.send_message([
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=part.function_call.name,
                                    response={"result": cleaned_result_data}
                                )
                            )
                        ])

                        # Extract final response after function execution
                        if (follow_up.candidates and
                            follow_up.candidates[0].content and
                            follow_up.candidates[0].content.parts):
                            for follow_part in follow_up.candidates[0].content.parts:
                                if follow_part.text:
                                    final_response += follow_part.text
                        else:
                            logger.warning("⚠️ Empty follow-up response after function call")

                    except Exception as follow_up_error:
                        logger.error(f"❌ Follow-up message failed: {follow_up_error}")
                        # This is a common Gemini 2.5 issue - treat as recoverable
                        final_response += f"\n[Tool executed but response processing incomplete: {execution_result.error if not execution_result.success else 'completed'}]"

            # Validate final response
            if not final_response.strip():
                raise ValueError("Empty final response generated (possible Gemini 2.5 cutoff)")

            logger.info(f"✅ Conversation processed successfully for {user_id} (attempt {attempt + 1})")
            logger.info(f"📊 Function calls processed: {function_calls_processed}")
            return final_response

        except Exception as e:
            logger.error(f"❌ Conversation processing failed (attempt {attempt + 1}): {e}")

            # Check if this is a recoverable error for Gemini 2.5 issues
            error_str = str(e).lower()
            recoverable_errors = [
                "empty response",
                "malformed response",
                "cutoff",
                "tool call",
                "function call",
                "follow-up",
                "response processing incomplete"
            ]

            is_recoverable = any(keyword in error_str for keyword in recoverable_errors)

            if is_recoverable and attempt < max_conversation_retries:
                logger.info("🔄 Recoverable error detected, will retry conversation processing")

                # If session recovery is enabled, try to recreate the session
                if session_recovery_enabled and attempt > 0:
                    logger.info(f"🔄 Attempting to recreate session for {user_id}")
                    try:
                        if session_key in active_chat_sessions:
                            del active_chat_sessions[session_key]
                        if session_key in session_tool_versions:
                            del session_tool_versions[session_key]
                        logger.info("🧹 Session cleared for recreation")
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Session cleanup warning: {cleanup_error}")

                continue  # Retry the conversation
            else:
                # Non-recoverable error or max retries reached
                if attempt == max_conversation_retries:
                    logger.error(f"💥 All conversation attempts failed for {user_id}")

                # Return a fallback response instead of failing completely
                fallback_response = "I apologize, but I'm experiencing some technical difficulties processing your request. Please try again, and I'll do my best to help you."
                logger.warning(f"🛡️ Returning fallback response for {user_id}")
                return fallback_response

    # This should never be reached due to the loop structure, but safety fallback
    return "I'm here and ready to help, though I may have encountered some processing issues."

async def _analyze_conversation_for_autonomic_tasks(
    user_message: str,
    aura_response: str,
    user_id: str,
    session_id: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Analyze conversation for potential autonomic tasks that could enhance future interactions
    
    Returns:
        List of (task_description, task_payload) tuples for autonomic processing
    """
    potential_tasks = []
    
    # Analyze for memory consolidation opportunities
    if len(user_message.split()) > 10 or len(aura_response.split()) > 20:
        potential_tasks.append((
            f"Analyze and consolidate conversation memory patterns for user {user_id}",
            {
                "task_type": "memory_consolidation",
                "user_message": user_message,
                "aura_response": aura_response,
                "user_id": user_id,
                "session_id": session_id,
                "conversation_length": len(user_message) + len(aura_response)
            }
        ))
    
    # Analyze for emotional pattern tracking
    emotional_keywords = [
        "feel", "emotion", "mood", "happy", "sad", "excited", "worried", 
        "anxious", "calm", "peaceful", "frustrated", "angry", "love"
    ]
    
    if any(keyword in user_message.lower() for keyword in emotional_keywords):
        potential_tasks.append((
            f"Deep emotional pattern analysis for user {user_id}",
            {
                "task_type": "emotional_analysis",
                "user_message": user_message,
                "user_id": user_id,
                "session_id": session_id,
                "analysis_scope": "emotional_patterns"
            }
        ))
    
    # Analyze for learning pattern optimization
    learning_keywords = [
        "learn", "understand", "explain", "teach", "show", "how to", 
        "what is", "why", "concept", "idea", "knowledge"
    ]
    
    if any(keyword in user_message.lower() for keyword in learning_keywords):
        potential_tasks.append((
            f"Optimize learning patterns and knowledge structure for user {user_id}",
            {
                "task_type": "learning_optimization",
                "user_message": user_message,
                "user_id": user_id,
                "session_id": session_id,
                "learning_context": "knowledge_acquisition"
            }
        ))
    
    # Analyze for background memory search and preparation
    if "remember" in user_message.lower() or "recall" in user_message.lower():
        potential_tasks.append((
            f"Proactive memory search and context preparation for user {user_id}",
            {
                "task_type": "proactive_memory_search",
                "query": user_message,
                "user_id": user_id,
                "session_id": session_id,
                "max_results": 15
            }
        ))
    
    # Analyze for relationship and context building
    if len(potential_tasks) == 0 and len(user_message.split()) > 5:
        # Default background task for context building
        potential_tasks.append((
            f"Background context analysis and relationship mapping for user {user_id}",
            {
                "task_type": "context_building",
                "user_message": user_message,
                "user_id": user_id,
                "session_id": session_id,
                "analysis_type": "relationship_mapping"
            }
        ))
    
    return potential_tasks
@app.post("/search")
async def search_memories(request: SearchRequest):
    """Search through conversation memories using Aura's internal MCP tools"""
    try:
        # Use Aura's internal memory search tools for comprehensive search
        # This includes video archives and unified memory search capabilities
        if aura_internal_tools:
            # Try the advanced search_all_memories first (includes video archives)
            try:
                advanced_result = await aura_internal_tools.execute_tool(
                    "aura.search_all_memories",
                    {
                        "query": request.query,
                        "user_id": request.user_id,
                        "max_results": request.n_results
                    }
                )

                if advanced_result and advanced_result.get("status") == "success":
                    memories = advanced_result.get("memories", [])
                    # Convert to expected frontend format
                    formatted_results = []
                    for memory in memories:
                        formatted_results.append({
                            "content": memory.get("content", ""),
                            "metadata": memory.get("metadata", {}),
                            "similarity": memory.get("similarity", 0.0)
                        })

                    logger.info(f"🔍 Advanced search found {len(formatted_results)} memories using video + active search")
                    return {
                        "results": formatted_results,
                        "query": request.query,
                        "total_found": len(formatted_results),
                        "search_type": "unified_memory_search",
                        "includes_video_archives": True
                    }

            except Exception as e:
                logger.warning(f"⚠️ Advanced search failed, falling back to basic search: {e}")

            # Fallback to basic memory search if advanced fails
            try:
                basic_result = await aura_internal_tools.execute_tool(
                    "aura.search_memories",
                    {
                        "query": request.query,
                        "user_id": request.user_id,
                        "n_results": request.n_results
                    }
                )

                if basic_result and basic_result.get("status") == "success":
                    memories = basic_result.get("memories", [])
                    logger.info(f"🔍 Basic search found {len(memories)} memories using active search")
                    return {
                        "results": memories,
                        "query": request.query,
                        "total_found": len(memories),
                        "search_type": "active_memory_search",
                        "includes_video_archives": False
                    }

            except Exception as e:
                logger.warning(f"⚠️ Basic MCP search failed, using direct persistence: {e}")

        # Final fallback to direct persistence service if MCP tools fail
        if conversation_persistence:
            results = await conversation_persistence.safe_search_conversations(
                query=request.query,
                user_id=request.user_id,
                n_results=request.n_results
            )

            logger.info(f"🔍 Direct persistence search found {len(results)} memories")
            return {
                "results": results,
                "query": request.query,
                "total_found": len(results),
                "search_type": "persistence_fallback",
                "includes_video_archives": False
            }
        else:
            logger.error("❌ Conversation persistence service not available")
            return {
                "results": [],
                "query": request.query,
                "total_found": 0,
                "search_type": "no_persistence_available",
                "includes_video_archives": False,
                "error": "Persistence service not initialized"
            }

    except Exception as e:
        logger.error(f"❌ Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotional-analysis/{user_id}")
async def get_emotional_analysis(
    user_id: str,
    period: str = "week",  # Options: hour, day, week, month, year, multi-year
    custom_days: Optional[int] = None
):
    """Get emotional pattern analysis with granular time periods"""
    try:
        # Convert period to days
        period_mapping = {
            "hour": 1/24,      # Last hour
            "day": 1,          # Last 24 hours
            "week": 7,         # Last 7 days
            "month": 30,       # Last 30 days
            "year": 365,       # Last year
            "multi-year": 1825  # Last 5 years
        }

        # Use custom days if provided, otherwise use period mapping
        days = custom_days if custom_days is not None else period_mapping.get(period, 7)

        if not vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")

        analysis = await vector_db.analyze_emotional_trends(user_id, days)
        analysis["period_type"] = period
        analysis["custom_days"] = custom_days

        return analysis

    except Exception as e:
        logger.error(f"❌ Failed to get emotional analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/{user_id}")
async def export_user_data(user_id: str, format: str = "json"):
    """Export user conversation history and patterns"""
    try:
        if not aura_file_system:
            raise HTTPException(status_code=500, detail="File system not initialized")

        export_path = await aura_file_system.export_conversation_history(user_id, format)
        return {"export_path": export_path, "message": "Export completed successfully"}

    except Exception as e:
        logger.error(f"❌ Failed to export user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user with thread-safe database access"""
    try:
        if not conversation_persistence:
            raise HTTPException(status_code=500, detail="Conversation persistence service not initialized")

        # Use the persistence service's thread-safe method
        result = await conversation_persistence.safe_get_chat_history(user_id, limit)
        return result

    except Exception as e:
        logger.error(f"❌ Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-history/{user_id}/{session_id}")
async def delete_chat_session(user_id: str, session_id: str):
    """Delete a specific chat session using enhanced database operations"""
    try:
        if not vector_db or not vector_db.conversations:
            raise HTTPException(status_code=500, detail="Vector database not properly initialized")

        # Use safe operation wrapper for database access
        async with vector_db._safe_operation("delete_chat_session"):
            # Get all messages for this session
            results = vector_db.conversations.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"session_id": {"$eq": session_id}}
                    ]
                },
                include=["documents", "metadatas"]
            )

            if results and results.get('ids'):
                # Delete all messages in this session
                vector_db.conversations.delete(ids=results['ids'])

                # Also remove from active sessions if present
                session_key = f"{user_id}_{session_id}"
                if session_key in active_chat_sessions:
                    del active_chat_sessions[session_key]
                if session_key in session_tool_versions:
                    del session_tool_versions[session_key]

                return {"message": f"Deleted session {session_id}", "deleted_count": len(results['ids'])}
            else:
                return {"message": "Session not found", "deleted_count": 0}

    except Exception as e:
        logger.error(f"❌ Failed to delete chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mcp/execute-tool")
async def mcp_execute_tool(request: ExecuteToolRequest):
    """Execute an MCP tool directly"""
    try:
        result = await execute_mcp_tool(
            tool_name=request.tool_name,
            arguments=request.arguments,
            user_id=request.user_id,
            aura_internal_tools=aura_internal_tools
        )
        return {"result": result}
    except Exception as e:
        logger.error(f"❌ MCP tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{user_id}")
async def clear_user_sessions(user_id: str):
    """Clear chat sessions for a user"""
    try:
        global active_chat_sessions
        sessions_cleared = 0

        # Find and remove all sessions for this user
        sessions_to_remove = [key for key in active_chat_sessions.keys() if key.startswith(f"{user_id}_")]

        for session_key in sessions_to_remove:
            del active_chat_sessions[session_key]
            sessions_cleared += 1

        logger.info(f"🧹 Cleared {sessions_cleared} chat sessions for user {user_id}")

        return {
            "message": f"Cleared {sessions_cleared} chat sessions for user {user_id}",
            "user_id": user_id,
            "sessions_cleared": sessions_cleared
        }

    except Exception as e:
        logger.error(f"❌ Failed to clear sessions for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/bridge-status")
async def get_mcp_bridge_status():
    """Get MCP-Gemini bridge status and statistics"""
    try:
        if not mcp_gemini_bridge:
            return {
                "status": "not_initialized",
                "message": "MCP-Gemini bridge is not initialized"
            }

        stats = mcp_gemini_bridge.get_execution_stats()
        available_functions = mcp_gemini_bridge.get_available_functions()

        return {
            "status": "active",
            "available_functions": len(available_functions),
            "execution_stats": stats,
            "sample_functions": available_functions[:5]  # Show first 5 functions
        }

    except Exception as e:
        logger.error(f"❌ Failed to get bridge status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/mcp/system-status")
async def get_mcp_system_status():
    """Get comprehensive MCP system status"""
    try:
        status = get_mcp_status()

        # Get detailed tool information if initialized
        if status["initialized"]:
            tools = await get_all_available_tools()

            # Group tools by server
            tools_by_server = {}
            for tool in tools:
                server = tool.get("server", "unknown")
                if server not in tools_by_server:
                    tools_by_server[server] = []
                tools_by_server[server].append({
                    "name": tool["name"],
                    "description": tool["description"][:100] + "..." if len(tool["description"]) > 100 else tool["description"]
                })

            status["tools_by_server"] = tools_by_server
            status["total_tools"] = len(tools)

        return status

    except Exception as e:
        logger.error(f"❌ Failed to get MCP system status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "initialized": False
        }

@app.get("/persistence/health")
async def get_persistence_health():
    """Get persistence layer health status"""
    try:
        if not conversation_persistence:
            return {
                "status": "not_initialized",
                "error": "Conversation persistence service not initialized",
                "timestamp": datetime.now().isoformat()
            }

        metrics = await conversation_persistence.get_persistence_metrics()

        # Import the health check class here to avoid import issues
        from conversation_persistence_service import PersistenceHealthCheck
        health_checker = PersistenceHealthCheck(conversation_persistence)
        health_status = await health_checker.check_health()

        return {
            "status": "healthy" if health_status["healthy"] else "unhealthy",
            "metrics": metrics,
            "health_check": health_status,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to get persistence health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/test/persistence")
async def test_persistence_reliability():
    """Test endpoint to validate chat persistence reliability"""
    try:
        if not conversation_persistence:
            return {
                "status": "error",
                "message": "Conversation persistence service not initialized"
            }

        # Create a test conversation exchange
        from main import ConversationMemory, ConversationExchange

        test_user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        test_session_id = str(uuid.uuid4())

        user_memory = ConversationMemory(
            user_id=test_user_id,
            message="This is a test message to validate persistence reliability",
            sender="user",
            session_id=test_session_id
        )

        aura_memory = ConversationMemory(
            user_id=test_user_id,
            message="This is a test response to validate that chat history saves correctly",
            sender="aura",
            session_id=test_session_id
        )

        test_exchange = ConversationExchange(
            user_memory=user_memory,
            ai_memory=aura_memory,
            session_id=test_session_id
        )

        # Test immediate persistence
        immediate_result = await conversation_persistence.persist_conversation_exchange_immediate(
            test_exchange,
            update_profile=False,
            timeout=3.0
        )

        # Verify the conversation was stored by searching for it
        search_result = await conversation_persistence.safe_search_conversations(
            query="test message to validate persistence",
            user_id=test_user_id,
            n_results=2
        )

        # Check chat history retrieval
        history_result = await conversation_persistence.safe_get_chat_history(test_user_id, 10)

        return {
            "status": "success",
            "test_results": {
                "immediate_persistence": {
                    "success": immediate_result["success"],
                    "duration_ms": immediate_result["duration_ms"],
                    "stored_components": immediate_result["stored_components"],
                    "method": immediate_result.get("method", "unknown")
                },
                "search_verification": {
                    "found_messages": len(search_result),
                    "search_successful": len(search_result) >= 2
                },
                "history_retrieval": {
                    "sessions_found": len(history_result.get("sessions", [])),
                    "retrieval_successful": len(history_result.get("sessions", [])) > 0
                }
            },
            "test_user_id": test_user_id,
            "persistence_validated": (
                immediate_result["success"] and
                len(search_result) >= 2 and
                len(history_result.get("sessions", [])) > 0
            ),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Persistence test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/memvid/status")
async def get_memvid_status():
    """Get memvid archival service status"""
    try:
        if not memvid_archival:
            return {
                "status": "not_initialized",
                "error": "Memvid archival service not initialized",
                "timestamp": datetime.now().isoformat()
            }

        # Get basic status info
        archives = await memvid_archival.list_archives()

        return {
            "status": "operational",
            "archives_count": len(archives),
            "archives": archives[:5],  # Show first 5 archives
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to get memvid status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/vector-db/health")
async def get_vector_db_health():
    """Get detailed vector database health information"""
    try:
        if not vector_db:
            return {
                "status": "not_initialized",
                "error": "Vector database not initialized",
                "timestamp": datetime.now().isoformat()
            }

        health_info = await vector_db.health_check()
        return health_info

    except Exception as e:
        logger.error(f"❌ Failed to get vector database health: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/autonomic/status")
async def get_autonomic_status():
    """Get comprehensive autonomic nervous system status"""
    try:
        autonomic_enabled = os.getenv('AUTONOMIC_ENABLED', 'true').lower() == 'true'
        
        if not autonomic_enabled:
            return {
                "status": "disabled",
                "message": "Autonomic nervous system is disabled in configuration",
                "timestamp": datetime.now().isoformat()
            }
        
        if not autonomic_system:
            return {
                "status": "not_initialized",
                "error": "Autonomic nervous system not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        status = autonomic_system.get_system_status()
        return {
            "status": "operational" if status["running"] else "stopped",
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get autonomic status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/autonomic/tasks/{user_id}")
async def get_user_autonomic_tasks(user_id: str, limit: int = 20):
    """Get autonomic tasks for a specific user"""
    try:
        if not autonomic_system:
            raise HTTPException(status_code=500, detail="Autonomic system not initialized")
        
        # Get completed tasks for the user
        user_tasks = []
        for task_id, task in autonomic_system.completed_tasks.items():
            if task.user_id == user_id:
                user_tasks.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "description": task.description,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "execution_time_ms": task.execution_time_ms,
                    "has_result": task.result is not None,
                    "has_error": task.error is not None
                })
        
        # Get active tasks for the user
        active_user_tasks = []
        for task_id, task in autonomic_system.active_tasks.items():
            if task.user_id == user_id:
                active_user_tasks.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "description": task.description,
                    "status": task.status.value,
                    "created_at": task.created_at.isoformat() if task.created_at else None,
                    "started_at": task.started_at.isoformat() if task.started_at else None
                })
        
        # Sort by creation time (newest first) and limit
        user_tasks.sort(key=lambda x: x["created_at"] or "", reverse=True)
        user_tasks = user_tasks[:limit]
        
        return {
            "user_id": user_id,
            "active_tasks": active_user_tasks,
            "completed_tasks": user_tasks,
            "total_active": len(active_user_tasks),
            "total_completed": len(user_tasks),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get user autonomic tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/autonomic/task/{task_id}")
async def get_autonomic_task_details(task_id: str):
    """Get detailed information about a specific autonomic task"""
    try:
        if not autonomic_system:
            raise HTTPException(status_code=500, detail="Autonomic system not initialized")
        
        # Check completed tasks first
        task = autonomic_system.completed_tasks.get(task_id)
        if not task:
            # Check active tasks
            task = autonomic_system.active_tasks.get(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task_details = {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "priority": task.priority.value,
            "description": task.description,
            "status": task.status.value,
            "user_id": task.user_id,
            "session_id": task.session_id,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "execution_time_ms": task.execution_time_ms,
            "payload": task.payload,
            "result": task.result,
            "error": task.error
        }
        
        return task_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get task details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autonomic/submit-task")
async def submit_autonomic_task(
    description: str,
    payload: Dict[str, Any],
    user_id: str,
    session_id: Optional[str] = None,
    force_offload: bool = False
):
    """Manually submit a task to the autonomic system"""
    try:
        if not autonomic_system:
            raise HTTPException(status_code=500, detail="Autonomic system not initialized")
        
        was_offloaded, task_id = await autonomic_system.submit_task(
            description=description,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
            force_offload=force_offload
        )
        
        if was_offloaded and task_id:
            return {
                "status": "submitted",
                "task_id": task_id,
                "message": f"Task {task_id} submitted to autonomic system",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "not_offloaded",
                "message": "Task did not meet criteria for autonomic processing",
                "force_offload_option": "Set force_offload=true to override",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit autonomic task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/autonomic/task/{task_id}/result")
async def get_autonomic_task_result(task_id: str, timeout: Optional[float] = None):
    """Get the result of an autonomic task, optionally waiting for completion"""
    try:
        if not autonomic_system:
            raise HTTPException(status_code=500, detail="Autonomic system not initialized")
        
        task = await autonomic_system.get_task_result(task_id, timeout)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "result": task.result,
            "error": task.error,
            "execution_time_ms": task.execution_time_ms,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get task result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autonomic/control/{action}")
async def control_autonomic_system(action: str):
    """Control autonomic system (start/stop/restart)"""
    try:
        if not autonomic_system:
            raise HTTPException(status_code=500, detail="Autonomic system not initialized")
        
        if action == "start":
            if autonomic_system._running:
                return {
                    "status": "already_running",
                    "message": "Autonomic system is already running",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                await autonomic_system.start()
                return {
                    "status": "started",
                    "message": "Autonomic system started successfully",
                    "timestamp": datetime.now().isoformat()
                }
        
        elif action == "stop":
            if not autonomic_system._running:
                return {
                    "status": "already_stopped",
                    "message": "Autonomic system is already stopped",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                await autonomic_system.stop()
                return {
                    "status": "stopped",
                    "message": "Autonomic system stopped successfully",
                    "timestamp": datetime.now().isoformat()
                }
        
        elif action == "restart":
            await autonomic_system.stop()
            await autonomic_system.start()
            return {
                "status": "restarted",
                "message": "Autonomic system restarted successfully",
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Use start, stop, or restart")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to control autonomic system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-db/optimize")
async def optimize_vector_db():
    """Trigger vector database optimization"""
    try:
        if not vector_db:
            raise HTTPException(status_code=500, detail="Vector database not initialized")

        # Perform SQLite optimization through the enhanced database
        async with vector_db._safe_operation("optimize_database"):
            import sqlite3
            sqlite_path = vector_db.persist_directory / "chroma.sqlite3"
            if sqlite_path.exists():
                conn = sqlite3.connect(str(sqlite_path))
                cursor = conn.cursor()
                cursor.execute("PRAGMA optimize")
                cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.commit()
                conn.close()

        return {
            "status": "success",
            "message": "Database optimization completed",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Failed to optimize vector database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Aura Backend Server...")
    logger.info("✨ Features: Vector DB, MCP Integration, Advanced State Management")
    logger.info("🔧 MCP Tools will be loaded on startup - check logs for available tools")
    logger.info("💡 To see available tools, ask Aura: 'What MCP tools do you have?'")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
