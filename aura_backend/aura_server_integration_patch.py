"""
Add these lines to your aura_server.py to enable Memvid-compatible archival
"""

# ==============================================================================
# ADD THIS IMPORT AT THE TOP (after the other imports, around line 15)
# ==============================================================================

# Add this after the other imports:
try:
    from aura_memvid_mcp_tools_compatible import add_compatible_memvid_tools
    MEMVID_COMPATIBLE_AVAILABLE = True
    logger.info("✅ Memvid-compatible tools loaded successfully")
except ImportError as e:
    MEMVID_COMPATIBLE_AVAILABLE = False
    logger.warning(f"⚠️ Memvid-compatible tools not available: {e}")

# ==============================================================================
# ADD THIS AFTER THE MCP SERVER IS CREATED (around line 850, after mcp = FastMCP(...))
# ==============================================================================

# Add this after: mcp = FastMCP(...)
if MEMVID_COMPATIBLE_AVAILABLE:
    add_compatible_memvid_tools(mcp)
    logger.info("🎯 Added Memvid-compatible archival tools to Aura MCP server")
    logger.info("📋 New tools: search_hybrid_memory, archive_old_conversations, import_knowledge_archive, get_memory_system_stats, list_archives")

# That's it! Your server now has revolutionary memory archival capabilities
