#!/usr/bin/env python3
"""
Test script for Aura MCP tool integration
"""

import asyncio
from mcp_integration import initialize_mcp_client, get_mcp_integration, shutdown_mcp_client
from mcp_tools import MCPToolParser

async def test_aura_mcp_integration():
    """Test Aura MCP integration and tools"""
    print("Testing Aura MCP integration...")
    
    # Initialize MCP client
    success = await initialize_mcp_client()
    
    if not success:
        print("⚠️ MCP client initialization issue - limited functionality available")
    else:
        print("✅ MCP client initialized successfully")
    
    # Get MCP client manager
    mcp_client = get_mcp_integration()
    
    # Get available tools
    available_tools = mcp_client.get_available_tools()
    print(f"Found {len(available_tools)} available tools")
    
    # Print details about each tool
    for i, tool in enumerate(available_tools):
        print(f"\nTool {i+1}: {tool['name']}")
        print(f"  Description: {tool['description']}")
        print(f"  Server: {tool['server']}")
        
        # Print parameters if available
        if tool.get('parameters'):
            print("  Parameters:")
            if isinstance(tool['parameters'], dict) and 'properties' in tool['parameters']:
                properties = tool['parameters'].get('properties', {})
                required = tool['parameters'].get('required', [])
                
                for param_name, param_details in properties.items():
                    is_required = param_name in required
                    param_type = param_details.get('type', 'any')
                    param_desc = param_details.get('description', '')
                    print(f"    - {param_name} ({param_type}{', required' if is_required else ''}): {param_desc}")
    
    # Test tool parser
    print("\nTesting MCP tool parser...")
    test_message = """
    Let me search for information about emotions using @mcp.tool("search_aura_memories", {"user_id": "test_user", "query": "emotional support"})
    
    And here's another tool:
    @mcp.tool("analyze_aura_emotional_patterns", 
    {
      "user_id": "test_user",
      "days": 7
    }
    )
    """
    
    parser = MCPToolParser(available_tools)
    tool_calls = parser.extract_tool_calls(test_message)
    
    print(f"Found {len(tool_calls)} tool calls in test message:")
    for call in tool_calls:
        print(f"  Tool: {call['tool_name']}")
        print(f"  Arguments: {call['arguments']}")
        print(f"  Exists: {call['exists']}")
        if not call['exists'] and 'qualified_name' in call:
            print(f"  Qualified name: {call['qualified_name']}")
        print("")
    
    # Shutdown MCP client
    await shutdown_mcp_client()
    print("✅ MCP client shutdown complete")

if __name__ == "__main__":
    asyncio.run(test_aura_mcp_integration())
