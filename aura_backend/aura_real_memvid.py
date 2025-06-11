"""
Real Aura + Memvid Integration (CHROMADB CONFLICT FIXED)
Uses actual memvid with QR-code video compression!

FIXED: ChromaDB instance conflict resolved - now reuses existing client
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Aura imports
import chromadb
from chromadb.config import Settings

# Placeholder classes for when real memvid isn't available
class _MemvidEncoderPlaceholder:
    def add_text(self, text): pass
    def add_pdf(self, path): pass
    def build_video(self, video_path, index_path, codec="h265", show_progress=True):
        return {"total_chunks": 0, "video_size_mb": 0, "total_frames": 0, "duration_seconds": 0, "fps": 0}

class _MemvidRetrieverPlaceholder:
    def __init__(self, video_file, index_file):
        self.video_file = video_file
        self.index_file = index_file
    def search_with_metadata(self, query, max_results=5): return []
    def get_stats(self): return {"video_file": self.video_file, "total_frames": 0, "fps": 0, "cache_size": 0}

class _MemvidChatPlaceholder:
    def __init__(self, video_file, index_file): pass

# REAL Memvid imports!
try:
    from memvid import MemvidEncoder, MemvidRetriever, MemvidChat
    REAL_MEMVID_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Real memvid imported successfully!")
except ImportError as e:
    REAL_MEMVID_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Real memvid not available: {e}, using placeholder classes")
    MemvidEncoder = _MemvidEncoderPlaceholder
    MemvidRetriever = _MemvidRetrieverPlaceholder
    MemvidChat = _MemvidChatPlaceholder

logger = logging.getLogger(__name__)

class AuraRealMemvid:
    """
    REAL Memvid integration with Aura (CHROMADB CONFLICT FIXED)
    Uses actual QR-code video compression for revolutionary memory storage!

    FIXED: Now accepts existing ChromaDB client to avoid instance conflicts
    """

    def __init__(self,
                 aura_chroma_path: str = "./aura_chroma_db",
                 memvid_video_path: str = "./memvid_videos",
                 active_memory_days: int = 30,
                 existing_chroma_client=None):  # NEW: Accept existing client

        self.aura_chroma_path = Path(aura_chroma_path)
        self.memvid_video_path = Path(memvid_video_path)
        self.active_memory_days = active_memory_days

        # Create video directory
        self.memvid_video_path.mkdir(exist_ok=True)

        # FIXED: Use existing ChromaDB client if provided, otherwise create new one carefully
        if existing_chroma_client is not None:
            logger.info("✅ Using existing ChromaDB client (conflict avoided)")
            self.chroma_client = existing_chroma_client
        else:
            try:
                # Try to connect to existing instance first
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.aura_chroma_path),
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info("✅ Connected to existing ChromaDB instance")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.warning(f"ChromaDB instance conflict detected: {e}")
                    logger.info("🔄 Attempting to use existing instance...")

                    # Try to get the existing client (this is a workaround)
                    try:
                        self.chroma_client = chromadb.PersistentClient(
                            path=str(self.aura_chroma_path),
                            settings=Settings(anonymized_telemetry=False, allow_reset=True)
                        )
                        logger.info("✅ Successfully connected after reset")
                    except Exception as e2:
                        logger.error(f"❌ Could not resolve ChromaDB conflict: {e2}")
                        raise RuntimeError(f"ChromaDB conflict: {e}. Please restart the application.")
                else:
                    raise e

        # Get existing collections
        try:
            self.conversations = self.chroma_client.get_collection("aura_conversations")
            self.emotional_patterns = self.chroma_client.get_collection("aura_emotional_patterns")
            logger.info("✅ Connected to existing Aura collections")
        except Exception as e:
            logger.warning(f"Could not connect to existing collections: {e}")
            try:
                # Create new collections if they don't exist
                self.conversations = self.chroma_client.get_or_create_collection("aura_conversations")
                self.emotional_patterns = self.chroma_client.get_or_create_collection("aura_emotional_patterns")
                logger.info("✅ Created new Aura collections")
            except Exception as e2:
                logger.error(f"❌ Failed to create collections: {e2}")
                raise

        # Load existing memvid video archives
        self.video_archives = {}
        self._load_existing_video_archives()

        archive_count = len(self.video_archives)
        memvid_status = "REAL" if REAL_MEMVID_AVAILABLE else "PLACEHOLDER"
        logger.info(f"🎥 {memvid_status} Memvid integration initialized with {archive_count} video archives")

    def _load_existing_video_archives(self):
        """Load existing memvid video archives"""
        if not REAL_MEMVID_AVAILABLE:
            logger.info("⚠️ Real memvid not available, skipping video archive loading")
            return

        for video_file in self.memvid_video_path.glob("*.mp4"):
            index_file = video_file.with_suffix(".json")
            if index_file.exists():
                archive_name = video_file.stem
                try:
                    self.video_archives[archive_name] = MemvidRetriever(
                        str(video_file), str(index_file)
                    )
                    logger.info(f"🎬 Loaded video archive: {archive_name}")
                except Exception as e:
                    logger.error(f"Failed to load video archive {archive_name}: {e}")

    def search_unified(self, query: str, user_id: str, max_results: int = 10) -> Dict:
        """
        Unified search across active ChromaDB and REAL memvid video archives
        FIXED: Better error handling for ChromaDB conflicts
        """
        results = {
            "query": query,
            "user_id": user_id,
            "active_results": [],
            "video_archive_results": [],
            "total_results": 0,
            "archive_type": "real_memvid_video" if REAL_MEMVID_AVAILABLE else "placeholder",
            "errors": []
        }

        # Search active memory (ChromaDB) with conflict protection
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = embedding_model.encode(query).tolist()

            active_search = self.conversations.query(
                query_embeddings=[query_embedding],
                n_results=max_results // 2,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"]
            )

            # More explicit checks for existence and structure of results
            docs_list = active_search.get("documents")
            meta_list = active_search.get("metadatas")
            dist_list = active_search.get("distances")

            if docs_list and isinstance(docs_list, list) and len(docs_list) > 0 and \
               docs_list[0] is not None and isinstance(docs_list[0], list):

                actual_documents = docs_list[0]
                actual_metadatas = meta_list[0] if meta_list and isinstance(meta_list, list) and len(meta_list) > 0 and meta_list[0] is not None and isinstance(meta_list[0], list) else []
                actual_distances = dist_list[0] if dist_list and isinstance(dist_list, list) and len(dist_list) > 0 and dist_list[0] is not None and isinstance(dist_list[0], list) else []

                for i, doc_content in enumerate(actual_documents):
                    distance = 0.0
                    if i < len(actual_distances):
                        distance = actual_distances[i]

                    metadata = {}
                    if i < len(actual_metadatas) and actual_metadatas[i] is not None:
                        metadata = actual_metadatas[i]

                    results["active_results"].append({
                        "text": doc_content,
                        "metadata": metadata,
                        "distance": distance,
                        "source": "active_memory",
                        "score": 1 - distance
                    })

        except Exception as e:
            error_msg = f"Error searching active memory: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

        # Search REAL memvid video archives (if available)
        if REAL_MEMVID_AVAILABLE:
            for archive_name, retriever in self.video_archives.items():
                try:
                    # Use real memvid search with metadata
                    video_results = retriever.search_with_metadata(query, top_k=max_results // 4)
                    for result in video_results:
                        results["video_archive_results"].append({
                            "text": result["text"],
                            "score": result["score"],
                            "source": f"video_archive:{archive_name}",
                            "chunk_id": result["chunk_id"],
                            "frame": result["frame"],
                            "video_file": archive_name + ".mp4"
                        })
                except Exception as e:
                    error_msg = f"Error searching video archive {archive_name}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
        else:
            results["errors"].append("Real memvid not available - video search disabled")

        results["total_results"] = len(results["active_results"]) + len(results["video_archive_results"])
        return results

    def archive_conversations_to_video(self, user_id: Optional[str] = None,
                                      codec: str = "h265") -> Dict:
        """
        Archive old conversations to REAL memvid video format!
        FIXED: Better error handling for ChromaDB operations
        """
        if not REAL_MEMVID_AVAILABLE:
            return {
                "error": "Real memvid not available",
                "archived_count": 0,
                "message": "Cannot create video archives without real memvid"
            }

        try:
            # Get old conversations from ChromaDB with conflict protection
            cutoff_date = datetime.now() - timedelta(days=self.active_memory_days)

            try:
                # Get all conversations for archival analysis
                all_conversations = self.conversations.get(
                    include=["documents", "metadatas"]
                )

                # Get IDs separately (ChromaDB quirk)
                all_ids = self.conversations.get(include=[])["ids"]
            except Exception as e:
                logger.error(f"ChromaDB access error during archival: {e}")
                return {
                    "error": f"Database access failed: {e}",
                    "archived_count": 0,
                    "suggestion": "Restart the application to resolve database conflicts"
                }

            if not all_conversations["documents"]:
                return {"archived_count": 0, "message": "No conversations to archive"}

            # Create REAL memvid encoder
            encoder = MemvidEncoder()

            conversations_to_archive = []
            ids_to_delete = []

            # Ensure documents exist before iterating
            documents_to_process = all_conversations.get("documents")
            if documents_to_process:
                for i, doc in enumerate(documents_to_process):
                    metadatas_list = all_conversations.get("metadatas")
                    metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                    doc_id = all_ids[i] if all_ids and i < len(all_ids) else f"doc_{i}"

                    # Check if this should be archived
                    timestamp_str = metadata.get('timestamp', '')
                    if timestamp_str and isinstance(timestamp_str, str):
                        try:
                            doc_timestamp = datetime.fromisoformat(timestamp_str)
                            if doc_timestamp < cutoff_date:
                                # Create rich memory context for video encoding
                                video_memory_text = f"""
AURA MEMORY ARCHIVE
==================
ID: {doc_id}
User: {metadata.get('user_id', 'unknown')}
Timestamp: {timestamp_str}
Emotional State: {metadata.get('emotion_name', 'none')}
Intensity: {metadata.get('emotion_intensity', 'none')}
Cognitive Focus: {metadata.get('cognitive_focus', 'none')}
Brainwave: {metadata.get('brainwave', 'none')}
Neurotransmitter: {metadata.get('neurotransmitter', 'none')}

CONVERSATION:
{doc}
==================
"""

                                encoder.add_text(video_memory_text.strip())

                                conversations_to_archive.append(doc_id)
                                ids_to_delete.append(doc_id)
                        except ValueError:
                            pass  # Skip invalid timestamps

            if not conversations_to_archive:
                return {"archived_count": 0, "message": "No old conversations found"}

            # Build REAL memvid video archive!
            archive_name = f"aura_video_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            video_path = self.memvid_video_path / f"{archive_name}.mp4"
            index_path = self.memvid_video_path / f"{archive_name}.json"

            logger.info(f"🎬 Creating video archive with {codec} codec...")

            build_stats = encoder.build_video(
                str(video_path),
                str(index_path),
                codec=codec,  # Use advanced video compression!
                show_progress=True
            )

            # Load new video archive
            self.video_archives[archive_name] = MemvidRetriever(
                str(video_path), str(index_path)
            )

            # Delete from ChromaDB with conflict protection
            if ids_to_delete:
                try:
                    self.conversations.delete(ids=ids_to_delete)
                    logger.info(f"✅ Deleted {len(ids_to_delete)} conversations from active memory")
                except Exception as e:
                    logger.error(f"⚠️ Failed to delete from ChromaDB: {e}")
                    # Continue anyway - the archive was created successfully

            logger.info(f"🎥 Archived {len(conversations_to_archive)} conversations to video: {archive_name}.mp4")

            return {
                "archived_count": len(conversations_to_archive),
                "archive_name": archive_name,
                "video_file": str(video_path),
                "video_codec": codec,
                "video_size_mb": build_stats.get("video_size_mb", 0),
                "compression_ratio": len(conversations_to_archive) / max(build_stats.get("video_size_mb", 1), 0.1),
                "total_frames": build_stats.get("total_frames", 0),
                "duration_seconds": build_stats.get("duration_seconds", 0),
                "archive_type": "real_memvid_video"
            }

        except Exception as e:
            logger.error(f"Error creating video archive: {e}")
            return {"error": str(e), "archived_count": 0}

    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics (CHROMADB CONFLICT SAFE)"""
        try:
            stats = {
                "memvid_type": "real_video_compression" if REAL_MEMVID_AVAILABLE else "placeholder",
                "real_memvid_available": REAL_MEMVID_AVAILABLE,
                "active_memory": {},
                "video_archives": {},
                "total_video_size_mb": 0,
                "chromadb_status": "connected"
            }

            # Safely get active memory stats
            try:
                stats["active_memory"] = {
                    "conversations": self.conversations.count() if self.conversations else 0,
                    "emotional_patterns": self.emotional_patterns.count() if self.emotional_patterns else 0
                }
            except Exception as e:
                logger.error(f"Error getting active memory stats: {e}")
                stats["active_memory"] = {"error": str(e)}
                stats["chromadb_status"] = "error"

            # Get video archive stats
            for name, retriever in self.video_archives.items():
                try:
                    archive_stats = retriever.get_stats()
                    stats["video_archives"][name] = archive_stats

                    # Calculate video file size
                    video_path = Path(archive_stats["video_file"])
                    if video_path.exists():
                        size_mb = video_path.stat().st_size / (1024 * 1024)
                        stats["total_video_size_mb"] += size_mb

                except Exception as e:
                    logger.error(f"Error getting stats for {name}: {e}")
                    stats["video_archives"][name] = {"error": str(e)}

            return stats

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "error": str(e),
                "memvid_type": "error",
                "real_memvid_available": REAL_MEMVID_AVAILABLE
            }

    def import_knowledge_to_video(self, source_path: str, archive_name: str,
                                 codec: str = "h265") -> Dict:
        """
        Import external documents into REAL memvid video archive
        Creates searchable MP4 files from documents!
        """
        if not REAL_MEMVID_AVAILABLE:
            return {
                "error": "Real memvid not available",
                "message": "Cannot create video archives without real memvid"
            }

        try:
            # Create real memvid encoder
            encoder = MemvidEncoder()
            source = Path(source_path)

            if source.is_file():
                if source.suffix.lower() == ".pdf":
                    logger.info(f"📄 Importing PDF to video: {source}")
                    encoder.add_pdf(str(source))
                elif source.suffix.lower() in [".txt", ".md"]:
                    logger.info(f"📝 Importing text to video: {source}")
                    with open(source, 'r', encoding='utf-8') as f:
                        encoder.add_text(f.read())
                else:
                    raise ValueError(f"Unsupported file type: {source.suffix}")

            elif source.is_dir():
                logger.info(f"📁 Importing directory to video: {source}")
                for file_path in source.rglob("*.txt"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        encoder.add_text(f.read())
                for file_path in source.rglob("*.pdf"):
                    encoder.add_pdf(str(file_path))
                for file_path in source.rglob("*.md"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        encoder.add_text(f.read())

            # Build REAL video archive
            video_path = self.memvid_video_path / f"{archive_name}.mp4"
            index_path = self.memvid_video_path / f"{archive_name}.json"

            logger.info(f"🎬 Building video archive with {codec} compression...")

            build_stats = encoder.build_video(
                str(video_path),
                str(index_path),
                codec=codec,
                show_progress=True
            )

            # Load video archive
            self.video_archives[archive_name] = MemvidRetriever(
                str(video_path), str(index_path)
            )

            logger.info(f"🎥 Created video knowledge base: {archive_name}.mp4")

            return {
                "archive_name": archive_name,
                "video_file": str(video_path),
                "video_codec": codec,
                "chunks_imported": build_stats.get("total_chunks", 0),
                "video_size_mb": build_stats.get("video_size_mb", 0),
                "compression_ratio": build_stats.get("total_chunks", 0) / max(build_stats.get("video_size_mb", 1), 0.1),
                "total_frames": build_stats.get("total_frames", 0),
                "duration_seconds": build_stats.get("duration_seconds", 0),
                "fps": build_stats.get("fps", 0),
                "source": str(source),
                "archive_type": "real_memvid_video"
            }

        except Exception as e:
            logger.error(f"Error creating video knowledge base: {e}")
            return {"error": str(e)}

    def create_memvid_chat(self, archive_name: str):
        """
        Create a MemvidChat instance for interactive conversation with a video archive
        """
        if not REAL_MEMVID_AVAILABLE:
            raise RuntimeError("Real memvid not available")

        if archive_name not in self.video_archives:
            raise ValueError(f"Video archive '{archive_name}' not found")

        retriever = self.video_archives[archive_name]
        video_file = retriever.video_file
        index_file = retriever.index_file

        return MemvidChat(video_file, index_file)

    def list_video_archives(self) -> List[Dict]:
        """List all available video archives with details"""
        archives = []

        for name, retriever in self.video_archives.items():
            try:
                stats = retriever.get_stats()
                video_path = Path(stats["video_file"])

                archive_info = {
                    "name": name,
                    "video_file": stats["video_file"],
                    "total_frames": stats.get("total_frames", 0),
                    "fps": stats.get("fps", 0),
                    "cache_size": stats.get("cache_size", 0),
                    "video_size_mb": video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0,
                    "can_play_as_video": REAL_MEMVID_AVAILABLE,  # The revolutionary feature!
                    "real_memvid": REAL_MEMVID_AVAILABLE
                }

                archives.append(archive_info)

            except Exception as e:
                logger.error(f"Error getting info for archive {name}: {e}")

        return archives

# Global instance for MCP integration (CHROMADB CONFLICT SAFE)
_aura_real_memvid = None

def get_aura_real_memvid(existing_chroma_client=None):
    """
    Get or create the real memvid system instance
    FIXED: Now properly manages shared ChromaDB client to prevent conflicts
    """
    global _aura_real_memvid
    if _aura_real_memvid is None:
        _aura_real_memvid = AuraRealMemvid(existing_chroma_client=existing_chroma_client)
    elif existing_chroma_client is not None and _aura_real_memvid.chroma_client != existing_chroma_client:
        # Reset instance if a different client is provided to ensure consistency
        logger.info("🔄 Resetting memvid instance to use provided ChromaDB client")
        _aura_real_memvid = AuraRealMemvid(existing_chroma_client=existing_chroma_client)
    return _aura_real_memvid

def reset_aura_real_memvid():
    """Reset the global instance (useful for resolving conflicts)"""
    global _aura_real_memvid
    _aura_real_memvid = None
