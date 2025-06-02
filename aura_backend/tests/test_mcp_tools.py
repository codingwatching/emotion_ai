#!/usr/bin/env python3
"""
Test script to verify MCP tools are properly loaded
"""

import asyncio
import logging
from mcp_integration import initialize_mcp_client, get_mcp_integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_tools():
    """Test MCP tool loading"""
    logger.info("🧪 Testing MCP tool loading...")
    
    # Initialize MCP client
    initialized = await initialize_mcp_client()
    if not initialized:
        logger.error("❌ Failed to initialize MCP client")
        return
    
    # Get MCP integration
    mcp_client = get_mcp_integration()
    
    # Get available tools
    tools = mcp_client.get_available_tools()
    
    logger.info(f"\n📦 Found {len(tools)} total MCP tools:")
    
    # Group by server
    tools_by_server = {}
    for tool in tools:
        server = tool.get('server', 'unknown')
        if server not in tools_by_server:
            tools_by_server[server] = []
        tools_by_server[server].append(tool)
    
    # Display tools by server
    for server, server_tools in tools_by_server.items():
        logger.info(f"\n🔧 {server} ({len(server_tools)} tools):")
        for tool in server_tools[:5]:  # Show first 5 tools per server
            logger.info(f"  - {tool['name']}: {tool['description'][:60]}...")
        if len(server_tools) > 5:
            logger.info(f"  ... and {len(server_tools) - 5} more tools")
    
    # Check specifically for aura-companion tools
    aura_tools = [t for t in tools if 'aura' in t['name'].lower() or t.get('server') == 'aura-companion']
    if aura_tools:
        logger.info(f"\n✅ Aura-specific tools found:")
        for tool in aura_tools:
            logger.info(f"  - {tool['name']} (server: {tool.get('server', 'unknown')})")
    else:
        logger.warning("\n⚠️ No Aura-specific tools found!")

if __name__ == "__main__":
    asyncio.run(test_mcp_tools())
