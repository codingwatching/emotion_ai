#!/usr/bin/env python3
"""
Surgical ChromaDB Repair - Reset only the corrupted conversations collection
while preserving working emotional patterns and other collections.

This targets the specific collection causing search failures without losing
the 443 working emotional pattern documents.
"""

import chromadb
from pathlib import Path
import logging
import sys
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def surgical_repair_conversations_collection():
    """Reset only the corrupted conversations collection"""
    
    try:
        logger.info("🔧 Starting surgical repair of conversations collection...")
        
        # Connect to the existing database
        client = chromadb.PersistentClient(path="./aura_chroma_db")
        
        # List all collections to see current state
        collections = client.list_collections()
        logger.info(f"📦 Found {len(collections)} collections:")
        
        working_collections = []
        for col in collections:
            try:
                count = col.count()
                logger.info(f"   ✅ {col.name}: {count} documents (working)")
                working_collections.append(col.name)
            except Exception as e:
                logger.error(f"   ❌ {col.name}: ERROR - {e}")
                
        # Check if conversations collection is the problem
        if "aura_conversations" not in [col.name for col in collections]:
            logger.warning("⚠️ aura_conversations collection not found, creating new one...")
        else:
            logger.info("🎯 Attempting to delete corrupted conversations collection...")
            try:
                client.delete_collection("aura_conversations")
                logger.info("✅ Deleted corrupted conversations collection")
            except Exception as e:
                logger.error(f"❌ Failed to delete conversations collection: {e}")
                logger.info("📝 Will create new collection anyway...")
        
        # Create fresh conversations collection
        logger.info("🆕 Creating fresh conversations collection...")
        conversations = client.create_collection(
            name="aura_conversations",
            metadata={"description": "Conversation history with semantic search"}
        )
        
        # Verify the new collection works
        test_count = conversations.count()
        logger.info(f"✅ New conversations collection created with {test_count} documents")
        
        # Test basic operations
        logger.info("🧪 Testing basic operations on new collection...")
        
        # Test add operation
        conversations.add(
            documents=["Test document to verify collection is working"],
            metadatas=[{"test": True, "timestamp": datetime.now().isoformat()}],
            ids=["test_doc_1"]
        )
        
        # Test query operation  
        results = conversations.query(
            query_texts=["test"],
            n_results=1
        )
        
        if results and results.get('documents'):
            logger.info("✅ New collection is fully functional!")
            
            # Clean up test document
            conversations.delete(ids=["test_doc_1"])
            logger.info("🧹 Cleaned up test document")
        else:
            logger.error("❌ New collection not working properly")
            return False
            
        # Final status report
        logger.info("\n📊 Repair Summary:")
        logger.info("   ✅ Conversations collection: Reset and functional")
        
        for col_name in working_collections:
            if col_name != "aura_conversations":
                logger.info(f"   ✅ {col_name}: Preserved")
                
        logger.info("\n🎉 Surgical repair completed successfully!")
        logger.info("💡 Search functionality should now work in the UI")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Surgical repair failed: {e}")
        return False

def verify_repair():
    """Verify that the repair was successful"""
    
    try:
        logger.info("\n🔍 Verifying repair...")
        
        client = chromadb.PersistentClient(path="./aura_chroma_db")
        collections = client.list_collections()
        
        conversations_found = False
        conversations_working = False
        
        for col in collections:
            if col.name == "aura_conversations":
                conversations_found = True
                try:
                    count = col.count()
                    # Try a basic query
                    results = col.query(query_texts=["test"], n_results=1)
                    conversations_working = True
                    logger.info(f"✅ Conversations collection: {count} documents, queries working")
                except Exception as e:
                    logger.error(f"❌ Conversations collection still broken: {e}")
                    
        if conversations_found and conversations_working:
            logger.info("🎯 Repair verification: SUCCESS")
            return True
        else:
            logger.error("❌ Repair verification: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False

if __name__ == "__main__":
    logger.info("🏥 ChromaDB Surgical Repair Tool")
    logger.info("   Target: Reset corrupted conversations collection only")
    logger.info("   Preserve: All other working collections and data")
    
    success = surgical_repair_conversations_collection()
    
    if success:
        verification_success = verify_repair()
        if verification_success:
            print("\n✅ REPAIR SUCCESSFUL")
            print("🔗 UI memory search should now work properly")
            print("📊 Emotional pattern data preserved") 
            sys.exit(0)
        else:
            print("\n⚠️ REPAIR COMPLETED BUT VERIFICATION FAILED")
            sys.exit(1)
    else:
        print("\n❌ REPAIR FAILED")
        sys.exit(1)
