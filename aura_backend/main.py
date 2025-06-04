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
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from contextlib import asynccontextmanager

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

# Import MCP integration
from mcp_system import (
    initialize_mcp_system,
    shutdown_mcp_system,
    get_mcp_status,
    get_mcp_bridge,
    # get_mcp_client,
    get_all_available_tools
)
from mcp_integration import (
    execute_mcp_tool,
    mcp_router,
)
from aura_internal_tools import AuraInternalTools

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

class AuraVectorDB:
    """Advanced vector database for Aura's memory and knowledge management"""

    def __init__(self, persist_directory: str = "./aura_chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Initialize vector database collections"""
        try:
            # Conversation memory collection
            self.conversations = self.client.get_or_create_collection(
                name="aura_conversations",
                metadata={"description": "Conversation history with semantic search"}
            )

            # Emotional patterns collection
            self.emotional_patterns = self.client.get_or_create_collection(
                name="aura_emotional_patterns",
                metadata={"description": "Historical emotional state patterns"}
            )

            # Cognitive patterns collection
            self.cognitive_patterns = self.client.get_or_create_collection(
                name="aura_cognitive_patterns",
                metadata={"description": "Cognitive focus and ASEKE component tracking"}
            )

            # Knowledge substrate collection
            self.knowledge_substrate = self.client.get_or_create_collection(
                name="aura_knowledge_substrate",
                metadata={"description": "Shared knowledge and insights"}
            )

            logger.info("✅ Vector database collections initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector collections: {e}")
            raise

    async def store_conversation(self, memory: ConversationMemory) -> str:
        """Store conversation memory with automatic embedding generation"""
        try:
            # Generate embedding if not provided
            if memory.embedding is None:
                memory.embedding = embedding_model.encode(memory.message).tolist()

            # Create unique ID
            if memory.timestamp is None:
                memory.timestamp = datetime.now()
            doc_id = f"{memory.user_id}_{memory.timestamp.isoformat()}_{uuid.uuid4().hex[:8]}"

            # Prepare metadata
            metadata = {
                "user_id": memory.user_id,
                "sender": memory.sender,
                "timestamp": memory.timestamp.isoformat(),
                "session_id": memory.session_id
            }

            # Add emotional state if present
            if memory.emotional_state:
                metadata.update({
                    "emotion_name": memory.emotional_state.name,
                    "emotion_intensity": memory.emotional_state.intensity.value,
                    "brainwave": memory.emotional_state.brainwave,
                    "neurotransmitter": memory.emotional_state.neurotransmitter
                })

            # Add cognitive state if present
            if memory.cognitive_state:
                metadata.update({
                    "cognitive_focus": memory.cognitive_state.focus.value,
                    "cognitive_description": memory.cognitive_state.description
                })

            # Store in vector database
            self.conversations.add(
                documents=[memory.message],
                embeddings=memory.embedding,
                metadatas=[metadata],
                ids=[doc_id]
            )

            logger.info(f"📝 Stored conversation memory: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Failed to store conversation memory: {e}")
            raise
    async def search_conversations(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Semantic search through conversation history"""
        try:
            # Generate query embedding
            query_embedding = embedding_model.encode(query).tolist()

            # Prepare where filter with proper typing for ChromaDB
            base_filter: Dict[str, Any] = {"user_id": {"$eq": user_id}}
            if where_filter:
                base_filter.update(where_filter)

            # Perform semantic search
            results = self.conversations.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=base_filter,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if (results is not None and
                results.get('documents') is not None and
                isinstance(results['documents'], list) and
                results['documents'] and
                results.get('metadatas') is not None and
                results['metadatas'] and
                results.get('distances') is not None and
                results['distances']):
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - results['distances'][0][i]  # Convert distance to similarity
                    })

            logger.info(f"🔍 Found {len(formatted_results)} relevant memories for query: {query}")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Failed to search conversations: {e}")
            return []
    async def store_emotional_pattern(self, emotional_state: EmotionalStateData, user_id: str) -> str:
        """Store emotional state pattern for analysis"""
        try:
            # Create embedding from emotional context
            emotion_text = f"{emotional_state.name} {emotional_state.description} {emotional_state.brainwave} {emotional_state.neurotransmitter}"
            embedding = embedding_model.encode(emotion_text).tolist()

            # Ensure timestamp is set
            if emotional_state.timestamp is None:
                emotional_state.timestamp = datetime.now()
            doc_id = f"emotion_{user_id}_{emotional_state.timestamp.isoformat()}_{uuid.uuid4().hex[:8]}"

            metadata = {
                "user_id": user_id,
                "emotion_name": emotional_state.name,
                "intensity": emotional_state.intensity.value,
                "brainwave": emotional_state.brainwave,
                "neurotransmitter": emotional_state.neurotransmitter,
                "timestamp": emotional_state.timestamp.isoformat(),
                "formula": emotional_state.formula
            }

            self.emotional_patterns.add(
                documents=[emotion_text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )

            logger.info(f"🎭 Stored emotional pattern: {emotional_state.name} ({emotional_state.intensity.value})")
            return doc_id

        except Exception as e:
            logger.error(f"❌ Failed to store emotional pattern: {e}")
            raise

    async def analyze_emotional_trends(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze emotional patterns over time"""
        try:
            # Get recent emotional data
            cutoff_date = datetime.now() - timedelta(days=days)

            results = self.emotional_patterns.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"timestamp": {"$gte": cutoff_date.isoformat()}}
                    ]
                },
                include=["metadatas"]
            )

            if not results['metadatas']:
                return {"message": "No emotional data found for analysis"}

            # Analyze patterns
            emotions = [str(meta['emotion_name']) for meta in results['metadatas']]
            intensities = [str(meta['intensity']) for meta in results['metadatas']]
            brainwaves = [str(meta['brainwave']) for meta in results['metadatas']]

            analysis = {
                "period_days": days,
                "total_entries": len(emotions),
                "dominant_emotions": self._get_top_items(emotions, 3),
                "intensity_distribution": self._get_distribution(intensities),
                "brainwave_patterns": self._get_distribution(brainwaves),
                "emotional_stability": self._calculate_stability(emotions),
                "recommendations": self._generate_emotional_recommendations(emotions, intensities)
            }

            logger.info(f"📊 Generated emotional analysis for user {user_id}")
            return analysis

        except Exception as e:
            logger.error(f"❌ Failed to analyze emotional trends: {e}")
            return {"error": str(e)}
    def _get_top_items(self, items: List[str], top_n: int) -> List[Tuple[str, int]]:
        """Get top N most frequent items"""
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(top_n)

    def _get_distribution(self, items: List[str]) -> Dict[str, int]:
        """Get distribution of items"""
        from collections import Counter
        return dict(Counter(items))

    def _calculate_stability(self, emotions: List[str]) -> float:
        """Calculate emotional stability score (0-1, higher = more stable)"""
        if len(emotions) <= 1:
            return 1.0

        from collections import Counter
        emotion_counts = Counter(emotions)
        entropy = -sum((count/len(emotions)) * np.log2(count/len(emotions))
                      for count in emotion_counts.values())
        max_entropy = np.log2(len(emotion_counts))

        # Normalize entropy to 0-1 and invert (higher = more stable)
        return 1 - (entropy / max_entropy if max_entropy > 0 else 0)

    def _generate_emotional_recommendations(self, emotions: List[str], intensities: List[str]) -> List[str]:
        """Generate emotional well-being recommendations"""
        recommendations = []

        # High intensity emotions
        high_intensity_count = intensities.count("High")
        if high_intensity_count > len(intensities) * 0.7:
            recommendations.append("Consider emotional regulation techniques - high intensity emotions detected")

        # Negative emotion patterns
        negative_emotions = ["Angry", "Sad", "Fear", "Disgust"]
        negative_count = sum(1 for emotion in emotions if emotion in negative_emotions)
        if negative_count > len(emotions) * 0.5:
            recommendations.append("Focus on positive emotional experiences and self-care activities")

        # Lack of variety
        unique_emotions = len(set(emotions))
        if unique_emotions < 3 and len(emotions) > 5:
            recommendations.append("Explore diverse experiences to expand emotional range")

        return recommendations or ["Emotional patterns appear balanced - continue current approach"]

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
            export_path = self.base_path / "aura_data/exports" / filename

            # This would integrate with the vector DB to get conversation history
            # For now, creating a placeholder structure
            # This should not be a placeholder anymore as the db is functional!!!
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

            # TODO: Could store recommendation in database for analysis or trigger gentle conversation adjustments
            # For now, we log the recommendation for monitoring emotional transition patterns

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

            self.vector_db.cognitive_patterns.add(
                documents=[focus_text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
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

# Initialize global components
vector_db = AuraVectorDB()
aura_file_system = AuraFileSystem()
state_manager = AuraStateManager(vector_db, aura_file_system)
aura_internal_tools = AuraInternalTools(vector_db, aura_file_system)

# Global MCP-Gemini bridge (will be initialized after MCP client startup)
mcp_gemini_bridge: Optional[MCPGeminiBridge] = None

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

**Aura's Internal MCP Tools (from aura-companion server):**
You have access to these built-in MCP tools for enhanced capabilities:

1. **search_aura_memories** - Search through conversation memories using semantic search
   - Parameters: user_id (string), query (string), n_results (int, default 5)
   - Use this to find relevant past conversations and emotional patterns

2. **analyze_aura_emotional_patterns** - Analyze emotional patterns over time
   - Parameters: user_id (string), days (int, default 7)
   - Provides insights into emotional stability, dominant emotions, and recommendations

3. **store_aura_conversation** - Store conversation memories with emotional/cognitive state
   - Parameters: user_id (string), message (string), sender (string), emotional_state (optional), cognitive_focus (optional)
   - Allows collaborative memory building

4. **get_aura_user_profile** - Retrieve user profile information
   - Parameters: user_id (string)
   - Access stored preferences and personalization data

5. **export_aura_user_data** - Export comprehensive user data
   - Parameters: user_id (string), format (string, default "json")
   - Enables data portability and backup

6. **query_aura_emotional_states** - Get info about your emotional state model
   - Returns details about the 22+ emotions, brainwaves, and neurotransmitters

7. **query_aura_aseke_framework** - Get details about your ASEKE cognitive architecture
   - Returns comprehensive information about all ASEKE components"""

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

    # Initialize Aura internal tools with global components
    global aura_internal_tools, mcp_gemini_bridge, global_tool_version

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

    yield

    # Shutdown
    logger.info("🛑 Shutting down Aura Backend...")
    await shutdown_mcp_system()
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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vector_db": "connected",
        "aura_file_system": "operational"
    }

@app.post("/conversation", response_model=ConversationResponse)
async def process_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Process conversation with MCP function calling and persistent chat context"""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        # Load user profile for context
        user_profile = await aura_file_system.load_user_profile(request.user_id)

        # Search relevant memories for context
        memory_context = ""
        if len(request.message.split()) > 3:
            relevant_memories = await vector_db.search_conversations(
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

        # Get or create persistent chat session
        session_key = f"{request.user_id}_{session_id}"

        # Check if we need to recreate session due to tool updates
        needs_new_session = session_key not in active_chat_sessions

        # Also check if existing session has outdated tool version
        if not needs_new_session and session_key in session_tool_versions:
            if session_tool_versions[session_key] < global_tool_version:
                needs_new_session = True
                logger.info(f"🔄 Session {session_key} has outdated tools (v{session_tool_versions[session_key]} < v{global_tool_version}), recreating...")

        if needs_new_session:
            # Create new chat session with MCP tools if available
            tools = []
            if mcp_gemini_bridge:
                gemini_tools = await mcp_gemini_bridge.convert_mcp_tools_to_gemini_functions()
                tools.extend(gemini_tools)
                logger.info(f"🔧 Added {len(gemini_tools)} MCP tools to chat session for {request.user_id}")

            # Create chat with system instruction and tools
            chat = client.chats.create(
                model=os.getenv('AURA_MODEL', 'gemini-2.5-flash-preview-05-20'),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=int(os.getenv('AURA_MAX_OUTPUT_TOKENS', '8192')),
                    tools=tools if tools else None
                )
            )
            active_chat_sessions[session_key] = chat
            session_tool_versions[session_key] = global_tool_version
            logger.info(f"💬 Created new chat session for {request.user_id} with {len(tools)} tools (v{global_tool_version})")
        else:
            chat = active_chat_sessions[session_key]
            logger.debug(f"💬 Using existing chat session for {request.user_id}")

        # Send message and handle function calls
        result = chat.send_message(request.message)

        # Process function calls if present
        final_response = ""
        if (result.candidates and result.candidates[0].content and
            result.candidates[0].content.parts):
            for part in result.candidates[0].content.parts:
                if part.text:
                    final_response += part.text
                elif hasattr(part, 'function_call') and part.function_call and mcp_gemini_bridge:
                    # Execute the function call through MCP bridge
                    logger.info(f"🔧 Executing function call: {part.function_call.name}")

                    execution_result = await mcp_gemini_bridge.execute_function_call(
                        part.function_call,
                        request.user_id
                    )

                    # Format and send function result back to model
                    # Send function result back to model
                    function_result_text = format_function_call_result_for_model(execution_result)

                    # *** NEW LINE ***
                    final_response += function_result_text + "\\n" # Append the raw tool result

                    # Send function result back to continue the conversation
                    follow_up = chat.send_message(
                        [types.Part(text=execution_result.result if execution_result.success else execution_result.error)]
                    )

                    # Extract final response after function execution
                    if (
                        follow_up.candidates and
                        follow_up.candidates[0].content is not None and
                        hasattr(follow_up.candidates[0].content, "parts") and
                        follow_up.candidates[0].content.parts
                    ):
                        for follow_part in follow_up.candidates[0].content.parts:
                            if follow_part.text:
                                final_response += follow_part.text

                    # Send function result back to continue the conversation
                    follow_up = chat.send_message([
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=part.function_call.name,
                                response={"result": execution_result.result if execution_result.success else execution_result.error}
                            )
                        )
                    ])

                    # Extract final response after function execution
                    if (
                        follow_up.candidates and
                        follow_up.candidates[0].content is not None and
                        hasattr(follow_up.candidates[0].content, "parts") and
                        follow_up.candidates[0].content.parts
                    ):
                        for follow_part in follow_up.candidates[0].content.parts:
                            if follow_part.text:
                                final_response += follow_part.text

        aura_response = final_response or "I'm here and ready to help!"

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
            emotional_state=user_emotional_state,  # Add user's emotional state
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

        # Store memories and update profile in background
        background_tasks.add_task(vector_db.store_conversation, user_memory)
        background_tasks.add_task(vector_db.store_conversation, aura_memory)

        if emotional_state_data:
            background_tasks.add_task(vector_db.store_emotional_pattern, emotional_state_data, request.user_id)

        # Store user's emotional pattern for analysis
        if user_emotional_state:
            background_tasks.add_task(vector_db.store_emotional_pattern, user_emotional_state, request.user_id)

        # Update user profile
        if user_profile is None:
            user_profile = {"name": request.user_id, "created_at": datetime.now().isoformat()}

        user_profile["last_interaction"] = datetime.now().isoformat()
        user_profile["total_messages"] = str(int(user_profile.get("total_messages", 0)) + 1)

        background_tasks.add_task(aura_file_system.save_user_profile, request.user_id, user_profile)

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
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/search")
async def search_memories(request: SearchRequest):
    """Search through conversation memories"""
    try:
        results = await vector_db.search_conversations(
            query=request.query,
            user_id=request.user_id,
            n_results=request.n_results
        )

        return {"results": results}

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
        export_path = await aura_file_system.export_conversation_history(user_id, format)
        return {"export_path": export_path, "message": "Export completed successfully"}

    except Exception as e:
        logger.error(f"❌ Failed to export user data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history for a user"""
    try:
        # Get recent conversations from vector DB
        results = vector_db.conversations.get(
            where={"user_id": user_id},
            limit=limit,
            include=["documents", "metadatas"]
        )

        if not results or not results.get('documents') or not isinstance(results['documents'], list):
            return {"sessions": [], "total": 0}

        # Group by session
        sessions = {}
        for i, doc in enumerate(results['documents']):
            metadata = results['metadatas'][i] if results.get('metadatas') and results['metadatas'] is not None else {}
            session_id = metadata.get('session_id', 'unknown')

            if session_id not in sessions:
                sessions[session_id] = {
                    "session_id": session_id,
                    "messages": [],
                    "start_time": metadata.get('timestamp', ''),
                    "last_time": metadata.get('timestamp', '')
                }

            sessions[session_id]["messages"].append({
                "content": doc,
                "sender": metadata.get('sender', 'unknown'),
                "timestamp": metadata.get('timestamp', ''),
                "emotion": metadata.get('emotion_name', 'Normal')
            })

            # Update session times
            if metadata.get('timestamp', '') < sessions[session_id]["start_time"]:
                sessions[session_id]["start_time"] = metadata.get('timestamp', '')
            if metadata.get('timestamp', '') > sessions[session_id]["last_time"]:
                sessions[session_id]["last_time"] = metadata.get('timestamp', '')

        # Convert to list and sort by last activity
        session_list = list(sessions.values())
        session_list.sort(key=lambda x: x["last_time"], reverse=True)

        return {
            "sessions": session_list,
            "total": len(session_list)
        }

    except Exception as e:
        logger.error(f"❌ Failed to get chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat-history/{user_id}/{session_id}")
async def delete_chat_session(user_id: str, session_id: str):
    """Delete a specific chat session"""
    try:
        # Get all messages for this session
        results = vector_db.conversations.get(
            where={
                "user_id": user_id,
                "session_id": session_id
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
