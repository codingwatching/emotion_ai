"""
Aura Internal Tools Integration
===============================

This module integrates Aura's internal capabilities directly into the main API,
eliminating the need for a separate MCP server process.
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AuraInternalTools:
    """Direct integration of Aura's internal tool capabilities"""

    def __init__(self, vector_db, file_system):
        self.vector_db = vector_db
        self.file_system = file_system
        self.tools = self._register_tools()
        logger.info("✅ Aura internal tools initialized")

    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register all Aura internal tools"""
        return {
            "aura.search_memories": {
                "name": "aura.search_memories",
                "description": "Search through Aura's conversation memories using semantic search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User ID"},
                        "query": {"type": "string", "description": "Search query"},
                        "n_results": {"type": "integer", "description": "Number of results", "default": 5}
                    },
                    "required": ["user_id", "query"]
                },
                "handler": self.search_memories
            },
            "aura.analyze_emotional_patterns": {
                "name": "aura.analyze_emotional_patterns",
                "description": "Analyze emotional patterns and trends over time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User ID"},
                        "days": {"type": "integer", "description": "Number of days to analyze", "default": 7}
                    },
                    "required": ["user_id"]
                },
                "handler": self.analyze_emotional_patterns
            },
            "aura.get_user_profile": {
                "name": "aura.get_user_profile",
                "description": "Retrieve user profile information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User ID"}
                    },
                    "required": ["user_id"]
                },
                "handler": self.get_user_profile
            },
            "aura.query_emotional_states": {
                "name": "aura.query_emotional_states",
                "description": "Get information about Aura's emotional state model",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "handler": self.query_emotional_states
            },
            "aura.query_aseke_framework": {
                "name": "aura.query_aseke_framework",
                "description": "Get details about Aura's ASEKE cognitive architecture",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                "handler": self.query_aseke_framework
            }
        }

    async def search_memories(self, user_id: str, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search through conversation memories"""
        try:
            results = await self.vector_db.search_conversations(
                query=query,
                user_id=user_id,
                n_results=n_results
            )

            return {
                "status": "success",
                "query": query,
                "user_id": user_id,
                "results_count": len(results),
                "memories": results
            }
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {"status": "error", "error": str(e)}

    async def analyze_emotional_patterns(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Analyze emotional patterns over time"""
        try:
            analysis = await self.vector_db.analyze_emotional_trends(user_id, days)

            return {
                "status": "success",
                "user_id": user_id,
                "analysis_period_days": days,
                "emotional_analysis": analysis
            }
        except Exception as e:
            logger.error(f"Failed to analyze emotional patterns: {e}")
            return {"status": "error", "error": str(e)}

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile"""
        try:
            profile = await self.file_system.load_user_profile(user_id)

            if profile is None:
                return {
                    "status": "not_found",
                    "user_id": user_id,
                    "message": "User profile not found"
                }

            return {
                "status": "success",
                "user_id": user_id,
                "profile": profile
            }
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return {"status": "error", "error": str(e)}

    async def query_emotional_states(self) -> Dict[str, Any]:
        """Get information about emotional states"""
        return {
            "status": "success",
            "emotional_system": {
                "total_emotions": "22+",
                "categories": [
                    "Basic emotions (Normal, Happy, Sad, Angry, Excited, Fear, Disgust, Surprise)",
                    "Complex emotions (Joy, Love, Peace, Creativity, DeepMeditation, Friendliness, Curiosity)",
                    "Combined emotions (Hope, Optimism, Awe, Remorse)",
                    "Social emotions (RomanticLove, PlatonicLove, ParentalLove)"
                ],
                "features": [
                    "Neurological correlations (Brainwaves, Neurotransmitters)",
                    "Mathematical formulas for emotional states",
                    "Intensity levels (Low, Medium, High)",
                    "Emotional component tracking",
                    "NTK (Neural Tensor Kernel) layer mapping"
                ]
            }
        }

    async def query_aseke_framework(self) -> Dict[str, Any]:
        """Get information about ASEKE framework"""
        return {
            "status": "success",
            "aseke_framework": {
                "framework_name": "ASEKE - Adaptive Socio-Emotional Knowledge Ecosystem",
                "components": {
                    "KS": "Knowledge Substrate - shared context and history",
                    "CE": "Cognitive Energy - focus and mental effort",
                    "IS": "Information Structures - ideas and concepts",
                    "KI": "Knowledge Integration - connecting understanding",
                    "KP": "Knowledge Propagation - sharing ideas",
                    "ESA": "Emotional State Algorithms - emotional influence",
                    "SDA": "Sociobiological Drives - social dynamics"
                }
            }
        }

    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available internal tools"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "server": "aura-internal",
                "parameters": tool["parameters"]
            }
            for name, tool in self.tools.items()
        ]

    def get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get tool definitions for the MCP bridge"""
        definitions = {}
        for name, tool in self.tools.items():
            definitions[name] = {
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        return definitions

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an internal tool"""
        if tool_name not in self.tools:
            # Try with aura. prefix
            if tool_name.startswith("aura."):
                tool_name = tool_name
            else:
                tool_name = f"aura.{tool_name}"

        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not found")

        handler = self.tools[tool_name]["handler"]
        return await handler(**arguments)
