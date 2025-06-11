#!/usr/bin/env python3
"""
Test script to verify memvid integration with Aura
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memvid_import():
    """Test that memvid can be imported correctly"""
    try:
        from memvid import MemvidEncoder, MemvidRetriever, MemvidChat
        logger.info("✅ Real memvid classes imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import real memvid: {e}")
        return False

def test_aura_memvid_integration():
    """Test that the Aura + memvid integration works"""
    try:
        from aura_real_memvid import AuraRealMemvid, REAL_MEMVID_AVAILABLE
        logger.info(f"✅ AuraRealMemvid imported, memvid available: {REAL_MEMVID_AVAILABLE}")
        
        if REAL_MEMVID_AVAILABLE:
            # Test basic initialization (without ChromaDB to avoid conflicts)
            logger.info("🎥 Testing memvid system initialization...")
            
            # Just test that the classes can be used
            from memvid import MemvidEncoder
            encoder = MemvidEncoder()
            logger.info("✅ MemvidEncoder can be instantiated")
            
            return True
        else:
            logger.warning("⚠️ Real memvid not available in AuraRealMemvid")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to test AuraRealMemvid integration: {e}")
        return False

def test_mcp_tools_integration():
    """Test that the MCP tools integration works"""
    try:
        from aura_memvid_mcp_tools_compatible_fixed import add_compatible_memvid_tools
        logger.info("✅ MCP tools integration imported successfully")
        
        # Test that the function exists and is callable
        if callable(add_compatible_memvid_tools):
            logger.info("✅ add_compatible_memvid_tools is callable")
            return True
        else:
            logger.error("❌ add_compatible_memvid_tools is not callable")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to test MCP tools integration: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("🧪 Testing Aura + Memvid Integration...")
    logger.info("=" * 50)
    
    tests = [
        ("Memvid Import", test_memvid_import),
        ("Aura+Memvid Integration", test_aura_memvid_integration), 
        ("MCP Tools Integration", test_mcp_tools_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
            status = "✅ PASSED" if results[test_name] else "❌ FAILED"
            logger.info(f"   {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"   ❌ FAILED with exception: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 INTEGRATION TEST SUMMARY:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Memvid integration is working!")
        return True
    else:
        logger.error(f"⚠️ {total - passed} tests failed. Integration needs fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
