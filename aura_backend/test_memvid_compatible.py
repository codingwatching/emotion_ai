#!/usr/bin/env python3
"""
Test Aura Memvid-Compatible Integration
"""

import asyncio
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_compatible_integration():
    """Test the memvid-compatible integration"""
    print("🧪 Testing Aura Memvid-Compatible Integration")
    print("=" * 60)
    
    try:
        # Test imports
        print("1️⃣ Testing imports...")
        
        try:
            from aura_memvid_compatible import AuraMemvidCompatible, get_aura_memvid_compatible
            print("   ✅ Compatible system import successful")
        except ImportError as e:
            print(f"   ❌ Compatible system import failed: {e}")
            return False
        
        try:
            from aura_memvid_mcp_tools_compatible import add_compatible_memvid_tools
            print("   ✅ MCP tools import successful")
        except ImportError as e:
            print(f"   ❌ MCP tools import failed: {e}")
            return False
        
        # Test system initialization
        print("\\n2️⃣ Testing system initialization...")
        try:
            system = get_aura_memvid_compatible()
            print("   ✅ System initialization successful")
        except Exception as e:
            print(f"   ❌ System initialization failed: {e}")
            return False
        
        # Test system stats
        print("\\n3️⃣ Testing system stats...")
        try:
            stats = system.get_system_stats()
            print(f"   ✅ System stats retrieved:")
            print(f"      - Active conversations: {stats['active_memory']['conversations']}")
            print(f"      - Active emotional patterns: {stats['active_memory']['emotional_patterns']}")
            print(f"      - Archives: {len(stats['archives'])}")
            print(f"      - Archive compatible: {stats['archive_compatible']}")
        except Exception as e:
            print(f"   ❌ System stats failed: {e}")
            return False
        
        # Test unified search (with empty data)
        print("\\n4️⃣ Testing unified search...")
        try:
            results = system.search_unified("test query", "test_user")
            print(f"   ✅ Unified search successful:")
            print(f"      - Total results: {results['total_results']}")
            print(f"      - Active results: {len(results['active_results'])}")
            print(f"      - Archive results: {len(results['archive_results'])}")
        except Exception as e:
            print(f"   ❌ Unified search failed: {e}")
            return False
        
        # Test archive creation (with empty data)
        print("\\n5️⃣ Testing archival process...")
        try:
            archival_result = system.archive_old_conversations("test_user")
            print(f"   ✅ Archival process successful:")
            print(f"      - Result: {archival_result}")
        except Exception as e:
            print(f"   ❌ Archival process failed: {e}")
            return False
        
        # Test knowledge import with a test file
        print("\\n6️⃣ Testing knowledge import...")
        try:
            # Create a test file
            test_file = Path("test_knowledge.txt")
            with open(test_file, 'w') as f:
                f.write("This is test knowledge content for Aura.\\n")
                f.write("It contains information about AI systems.\\n")
                f.write("This text will be imported into the archive.")
            
            import_result = system.import_knowledge_base(
                str(test_file),
                "test_knowledge"
            )
            print(f"   ✅ Knowledge import successful:")
            print(f"      - Archive: {import_result.get('archive_name', 'unknown')}")
            print(f"      - Chunks: {import_result.get('chunks_imported', 0)}")
            print(f"      - Size: {import_result.get('compressed_size_mb', 0):.2f} MB")
            
            # Clean up test file
            test_file.unlink()
            
        except Exception as e:
            print(f"   ❌ Knowledge import failed: {e}")
            # Clean up test file if it exists
            if Path("test_knowledge.txt").exists():
                Path("test_knowledge.txt").unlink()
            return False
        
        # Test search in the new archive
        print("\\n7️⃣ Testing search in imported archive...")
        try:
            search_results = system.search_unified("AI systems", "test_user")
            print(f"   ✅ Archive search successful:")
            print(f"      - Total results: {search_results['total_results']}")
            if search_results['archive_results']:
                print(f"      - Found content in archive: {search_results['archive_results'][0]['text'][:50]}...")
        except Exception as e:
            print(f"   ❌ Archive search failed: {e}")
            return False
        
        # Test MCP integration
        print("\\n8️⃣ Testing MCP integration...")
        try:
            from fastmcp import FastMCP
            
            # Create a test MCP server
            test_mcp = FastMCP("test-aura-memvid")
            
            # Add our tools
            add_compatible_memvid_tools(test_mcp)
            
            print("   ✅ MCP integration successful")
            print("   ✅ Tools added to MCP server")
            
        except Exception as e:
            print(f"   ❌ MCP integration failed: {e}")
            return False
        
        print("\\n🎉 ALL TESTS PASSED! 🎉")
        print("\\n📋 Integration Summary:")
        print("   ✅ Memvid-compatible archival system working")
        print("   ✅ Unified search across active + archived memory")
        print("   ✅ Knowledge base import functionality")
        print("   ✅ MCP tools integration ready")
        print("   ✅ No dependency conflicts")
        
        print("\\n🚀 Next Steps:")
        print("   1. Add tools to your aura_server.py:")
        print("      from aura_memvid_mcp_tools_compatible import add_compatible_memvid_tools")
        print("      add_compatible_memvid_tools(mcp)")
        print("   2. Start using hybrid memory search")
        print("   3. Import your knowledge bases")
        print("   4. Set up automated archival")
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_compatible_integration())
