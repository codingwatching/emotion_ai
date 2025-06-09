"""
Aura Internal Memvid Management Tools
===================================

Internal tools that allow Aura to directly manage its own memvid system.
These are different from the MCP tools - these are for Aura's internal use.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# Import the real memvid integration
try:
    from aura_real_memvid import get_aura_real_memvid, REAL_MEMVID_AVAILABLE
    INTERNAL_MEMVID_AVAILABLE = True
except ImportError:
    INTERNAL_MEMVID_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuraInternalMemvidTools:
    """
    Internal tools for Aura to manage its own memvid system
    These are NOT MCP tools - these are for Aura's direct use
    """
    
    def __init__(self, vector_db_client=None):
        self.vector_db_client = vector_db_client
        self.memvid_system = None
        
        if INTERNAL_MEMVID_AVAILABLE:
            try:
                # Initialize with existing ChromaDB client to avoid conflicts
                self.memvid_system = get_aura_real_memvid(existing_chroma_client=vector_db_client)
                logger.info("✅ Aura internal memvid tools initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize internal memvid tools: {e}")
                self.memvid_system = None
        else:
            logger.warning("⚠️ Real memvid not available for internal tools")
    
    async def list_video_archives(self) -> Dict[str, Any]:
        """
        Internal tool: List all video archives available to Aura
        Returns comprehensive information about video memory archives
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available",
                "archives": [],
                "total_archives": 0
            }
        
        try:
            archives = self.memvid_system.list_video_archives()
            
            result = {
                "status": "success", 
                "archives": archives,
                "total_archives": len(archives),
                "total_size_mb": sum(archive.get("video_size_mb", 0) for archive in archives),
                "compression_technology": "Real QR-code video compression",
                "searchable": True
            }
            
            logger.info(f"📋 Listed {len(archives)} video archives for Aura")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to list video archives: {e}")
            return {
                "status": "error",
                "message": str(e),
                "archives": [],
                "total_archives": 0
            }
    
    async def search_all_memories(self, query: str, user_id: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Internal tool: Search across ALL memory systems (active + video archives)
        This is Aura's unified memory search capability
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available",
                "results": [],
                "total_results": 0
            }
        
        try:
            # Perform unified search
            search_results = self.memvid_system.search_unified(
                query=query,
                user_id=user_id,
                max_results=max_results
            )
            
            # Format for internal use
            result = {
                "status": "success",
                "query": query,
                "user_id": user_id,
                "total_results": search_results["total_results"],
                "active_memory_results": len(search_results["active_results"]),
                "video_archive_results": len(search_results["video_archive_results"]),
                "all_results": search_results["active_results"] + search_results["video_archive_results"],
                "search_technology": "Unified vector + video search",
                "errors": search_results.get("errors", [])
            }
            
            logger.info(f"🔍 Unified search completed: {result['total_results']} results for '{query}'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to search all memories: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": [],
                "total_results": 0
            }
    
    async def archive_old_conversations(self, user_id: Optional[str] = None, codec: str = "h264") -> Dict[str, Any]:
        """
        Internal tool: Archive old conversations to video format
        Aura can use this to manage its own memory efficiently
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available",
                "archived_count": 0
            }
        
        try:
            # Archive conversations to video
            result = self.memvid_system.archive_conversations_to_video(
                user_id=user_id,
                codec=codec
            )
            
            if "error" in result:
                return {
                    "status": "error",
                    "message": result["error"],
                    "archived_count": 0
                }
            
            # Format success response
            response = {
                "status": "success",
                "archived_count": result.get("archived_count", 0),
                "archive_name": result.get("archive_name", "unknown"),
                "video_file": result.get("video_file", ""),
                "video_size_mb": result.get("video_size_mb", 0),
                "compression_ratio": result.get("compression_ratio", 0),
                "technology": "QR-code video compression",
                "codec_used": codec
            }
            
            logger.info(f"🎬 Archived {response['archived_count']} conversations to video")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to archive conversations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "archived_count": 0
            }
    
    async def import_knowledge(self, source_path: str, archive_name: str, codec: str = "h264") -> Dict[str, Any]:
        """
        Internal tool: Import external knowledge into video archives
        Aura can use this to expand its knowledge base
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available"
            }
        
        try:
            # Import knowledge to video
            result = self.memvid_system.import_knowledge_to_video(
                source_path=source_path,
                archive_name=archive_name,
                codec=codec
            )
            
            if "error" in result:
                return {
                    "status": "error",
                    "message": result["error"]
                }
            
            # Format success response
            response = {
                "status": "success",
                "archive_name": result.get("archive_name", archive_name),
                "chunks_imported": result.get("chunks_imported", 0),
                "video_size_mb": result.get("video_size_mb", 0),
                "compression_ratio": result.get("compression_ratio", 0),
                "source_file": source_path,
                "technology": "QR-code video compression",
                "codec_used": codec
            }
            
            logger.info(f"📚 Imported knowledge to video archive: {archive_name}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to import knowledge: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Internal tool: Get comprehensive memory system statistics
        Aura can use this to understand its own memory state
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available",
                "statistics": {}
            }
        
        try:
            # Get system stats
            stats = self.memvid_system.get_system_stats()
            
            # Add additional analysis
            archives = self.memvid_system.list_video_archives()
            
            result = {
                "status": "success",
                "memory_type": stats.get("memvid_type", "unknown"),
                "real_memvid_available": stats.get("real_memvid_available", False),
                "active_memory": stats.get("active_memory", {}),
                "video_archives": {
                    "count": len(archives),
                    "total_size_mb": stats.get("total_video_size_mb", 0),
                    "archives": archives
                },
                "technology": "Real QR-code video compression",
                "chromadb_status": stats.get("chromadb_status", "unknown"),
                "efficiency_metrics": {
                    "compression_active": True,
                    "searchable_videos": True,
                    "unified_search": True
                }
            }
            
            logger.info("📊 Generated memory system statistics for Aura")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory statistics: {e}")
            return {
                "status": "error",
                "message": str(e),
                "statistics": {}
            }
    
    async def create_knowledge_summary(self, archive_name: str, max_entries: int = 10) -> Dict[str, Any]:
        """
        Internal tool: Create a summary of what's in a video archive
        Aura can use this to understand its own knowledge
        """
        if not self.memvid_system:
            return {
                "status": "error",
                "message": "Memvid system not available"
            }
        
        try:
            # Check if archive exists
            archives = self.memvid_system.list_video_archives()
            target_archive = None
            
            for archive in archives:
                if archive["name"] == archive_name:
                    target_archive = archive
                    break
            
            if not target_archive:
                return {
                    "status": "error",
                    "message": f"Archive '{archive_name}' not found"
                }
            
            # Search for general content to get a sample
            sample_search = self.memvid_system.search_unified(
                query="content knowledge information",
                user_id="aura_internal",  # Special user ID for Aura's internal searches
                max_results=max_entries
            )
            
            # Filter for this specific archive
            archive_results = [
                result for result in sample_search.get("video_archive_results", [])
                if result.get("source", "").startswith(f"video_archive:{archive_name}")
            ]
            
            result = {
                "status": "success",
                "archive_name": archive_name,
                "total_frames": target_archive.get("total_frames", 0),
                "video_size_mb": target_archive.get("video_size_mb", 0),
                "sample_content": [
                    {
                        "text_preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                        "relevance_score": result.get("score", 0),
                        "frame": result.get("frame", 0)
                    }
                    for result in archive_results[:max_entries]
                ],
                "content_count": len(archive_results),
                "technology": "QR-code compressed video"
            }
            
            logger.info(f"📖 Created knowledge summary for archive: {archive_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to create knowledge summary: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

# Global instance
_aura_internal_memvid_tools = None

def get_aura_internal_memvid_tools(vector_db_client=None):
    """Get or create the internal memvid tools instance"""
    global _aura_internal_memvid_tools
    if _aura_internal_memvid_tools is None:
        _aura_internal_memvid_tools = AuraInternalMemvidTools(vector_db_client)
    return _aura_internal_memvid_tools

def reset_aura_internal_memvid_tools():
    """Reset the global instance"""
    global _aura_internal_memvid_tools
    _aura_internal_memvid_tools = None
