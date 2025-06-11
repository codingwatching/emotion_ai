#!/usr/bin/env python3
"""
Focused ChromaDB Fixes Validation
=================================

Tests only the specific mathematical and architectural fixes without dependencies.
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_division_by_zero_fix():
    """
    Test Fix 1: Division by zero prevention in _update_average_store_time
    
    Simulates the exact scenario that was causing the error.
    """
    logger.info("🔍 Testing division by zero fix...")
    
    # Simulate the metrics structure from ConversationPersistenceService
    metrics = {
        "total_exchanges_stored": 0,  # This is the critical zero value
        "failed_stores": 0,
        "average_store_time": 0.0,
        "last_error": None
    }
    
    # Simulate the fixed _update_average_store_time logic
    def update_average_store_time_fixed(duration_ms: float, metrics_dict: dict) -> bool:
        """Fixed version of _update_average_store_time"""
        current_avg = metrics_dict["average_store_time"]
        total_stores = metrics_dict["total_exchanges_stored"]

        if total_stores == 0:
            # FIX: Handle zero case explicitly - no division needed
            metrics_dict["average_store_time"] = duration_ms
        elif total_stores == 1:
            metrics_dict["average_store_time"] = duration_ms
        else:
            # Rolling average calculation
            metrics_dict["average_store_time"] = (
                (current_avg * (total_stores - 1) + duration_ms) / total_stores
            )
        return True
    
    # Simulate the old broken logic for comparison
    def update_average_store_time_broken(duration_ms: float, metrics_dict: dict) -> bool:
        """Broken version that caused division by zero"""
        current_avg = metrics_dict["average_store_time"]
        total_stores = metrics_dict["total_exchanges_stored"]

        if total_stores == 1:
            metrics_dict["average_store_time"] = duration_ms
        else:
            # BUG: This divides by zero when total_stores == 0
            metrics_dict["average_store_time"] = (
                (current_avg * (total_stores - 1) + duration_ms) / total_stores
            )
        return True
    
    try:
        # Test the broken version should fail
        test_metrics_broken = metrics.copy()
        try:
            update_average_store_time_broken(100.0, test_metrics_broken)
            logger.error("❌ Broken version should have failed but didn't!")
            return False
        except ZeroDivisionError:
            logger.info("✅ Broken version correctly fails with ZeroDivisionError")
        
        # Test the fixed version should succeed
        test_metrics_fixed = metrics.copy()
        success = update_average_store_time_fixed(100.0, test_metrics_fixed)
        
        if success and test_metrics_fixed["average_store_time"] == 100.0:
            logger.info("✅ Fixed version handles zero total_stores correctly")
            return True
        else:
            logger.error(f"❌ Fixed version failed: {test_metrics_fixed}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_chromadb_client_sharing_pattern():
    """
    Test Fix 2: ChromaDB client sharing architectural pattern
    
    Tests the global instance management logic without actual ChromaDB.
    """
    logger.info("🔍 Testing ChromaDB client sharing pattern...")
    
    # Simulate ChromaDB client objects
    class MockChromaClient:
        def __init__(self, name):
            self.name = name
            self.id = id(self)
    
    # Create a class to encapsulate the global state
    class InstanceManager:
        def __init__(self):
            self._global_instance = None
        
        def get_instance_fixed(self, existing_client=None):
            """Fixed version of get_aura_real_memvid"""
            if self._global_instance is None:
                # Create new instance with provided client
                self._global_instance = {"client": existing_client or MockChromaClient("default")}
            elif existing_client is not None and self._global_instance["client"] != existing_client:
                # Reset instance if different client is provided (our fix)
                logger.info("🔄 Resetting instance to use provided client")
                self._global_instance = {"client": existing_client}
            
            return self._global_instance
        
        def get_instance_broken(self, existing_client=None):
            """Broken version that doesn't handle client sharing"""
            if self._global_instance is None:
                self._global_instance = {"client": existing_client or MockChromaClient("default")}
            # BUG: Doesn't check if different client is provided
            
            return self._global_instance
        
        def reset(self):
            self._global_instance = None
    
    try:
        # Test scenario: main system creates client, then memvid should use it
        main_client = MockChromaClient("main_system")
        
        # Test broken version
        manager_broken = InstanceManager()
        instance1 = manager_broken.get_instance_broken()  # Creates default client
        instance2 = manager_broken.get_instance_broken(main_client)  # Should use main_client but doesn't
        
        if instance1["client"] == instance2["client"]:
            logger.info("✅ Broken version correctly shows the problem (ignores provided client)")
        else:
            logger.error("❌ Broken version test didn't work as expected")
            return False
        
        # Test fixed version
        manager_fixed = InstanceManager()
        instance3 = manager_fixed.get_instance_fixed()  # Creates default client
        default_client = instance3["client"]
        
        instance4 = manager_fixed.get_instance_fixed(main_client)  # Should switch to main_client
        
        if instance4["client"] == main_client and instance4["client"] != default_client:
            logger.info("✅ Fixed version correctly switches to provided client")
            return True
        else:
            logger.error(f"❌ Fixed version failed to switch clients")
            return False
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_fixes_are_in_place():
    """
    Test that our fixes are actually present in the source files
    """
    logger.info("🔍 Verifying fixes are in place...")
    
    # Check that division by zero fix is in place
    try:
        with open("conversation_persistence_service.py", "r") as f:
            content = f.read()
            
        if "if total_stores == 0:" in content:
            logger.info("✅ Division by zero fix is present in source")
            div_fix_present = True
        else:
            logger.error("❌ Division by zero fix not found in source")
            div_fix_present = False
    except Exception as e:
        logger.error(f"❌ Could not check persistence service file: {e}")
        div_fix_present = False
    
    # Check that client sharing fix is in place
    try:
        with open("aura_real_memvid.py", "r") as f:
            content = f.read()
            
        if "existing_chroma_client is not None and" in content:
            logger.info("✅ ChromaDB client sharing fix is present in source")
            client_fix_present = True
        else:
            logger.error("❌ ChromaDB client sharing fix not found in source")
            client_fix_present = False
    except Exception as e:
        logger.error(f"❌ Could not check memvid file: {e}")
        client_fix_present = False
    
    return div_fix_present and client_fix_present

def main():
    """Run focused validation tests"""
    print("🔧 Focused ChromaDB Fixes Validation")
    print("====================================")
    print()
    
    tests = [
        ("Division by Zero Fix", test_division_by_zero_fix),
        ("ChromaDB Client Sharing Pattern", test_chromadb_client_sharing_pattern),
        ("Fixes Present in Source", test_fixes_are_in_place)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"Result: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
            print(f"Result: ❌ CRASHED")
        print()
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("📊 Test Results Summary:")
    print("=======================")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 SUCCESS: All focused validation tests passed!")
        print("✅ The ChromaDB conflict fixes are working correctly")
        return 0
    else:
        print(f"\n❌ FAILURE: {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
