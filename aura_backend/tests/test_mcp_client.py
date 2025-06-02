#!/usr/bin/env python3
"""
Test script for Aura's MCP Client functionality
"""

import asyncio
import logging
from mcp_client import AuraMCPClient, AuraMCPIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Test MCP client functionality"""
    print("🧪 Testing Aura MCP Client Integration")
    print("=" * 50)
    
    # Create MCP client
    client = AuraMCPClient("mcp_client_config.json")
    integration = AuraMCPIntegration(client)
    
    try:
        # Start the client
        print("\n1. Starting MCP Client...")
        await client.start()
        
        # Wait a moment for connections
        await asyncio.sleep(2)
        
        # Get capabilities
        print("\n2. Checking MCP Capabilities...")
        capabilities = await integration.get_available_capabilities()
        print(f"   Connected servers: {capabilities['connected_servers']}/{capabilities['total_servers']}")
        print(f"   Available tools: {capabilities['available_tools']}")
        
        # List available tools
        print("\n3. Available MCP Tools:")
        tools = await client.list_all_tools()
        for tool_name, info in tools.items():
            if info['connected']:
                print(f"   ✅ {tool_name}: {info['description']}")
            else:
                print(f"   ❌ {tool_name}: {info['description']} (not connected)")
        
        # Test SQLite if available
        if any('sqlite' in name for name in tools):
            print("\n4. Testing SQLite Query...")
            try:
                result = await integration.search_database(
                    "SELECT name FROM sqlite_master WHERE type='table' LIMIT 5;"
                )
                print(f"   Result: {result}")
            except Exception as e:
                print(f"   Error: {e}")
        
        # Show call history
        print("\n5. Call History:")
        history = client.get_call_history(5)
        for call in history:
            print(f"   - {call['timestamp']}: {call['tool']} on {call['server']}")
            if not call['success']:
                print(f"     Error: {call['error']}")
        
        print("\n✅ MCP Client test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    
    finally:
        # Stop the client
        print("\n6. Stopping MCP Client...")
        await client.stop()
        print("   ✅ Client stopped")

if __name__ == "__main__":
    asyncio.run(main())
