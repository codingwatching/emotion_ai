"""
Emergency ChromaDB Recovery System
==================================

This module provides emergency recovery capabilities for ChromaDB when it becomes
completely unresponsive due to compaction failures or corruption issues.
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import asyncio

logger = logging.getLogger(__name__)

class EmergencyDBRecovery:
    """Handles emergency recovery of ChromaDB when standard recovery fails"""

    def __init__(self, db_path: str = "./aura_mcp_chroma_db"):
        self.db_path = Path(db_path)
        self.backup_root = Path("./chromadb_emergency_backups")
        self.backup_root.mkdir(exist_ok=True)

    def create_emergency_backup(self) -> Optional[Path]:
        """Create emergency backup before attempting recovery"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_root / f"emergency_backup_{timestamp}"

            if self.db_path.exists():
                logger.info(f"🚨 Creating emergency backup: {backup_path}")
                shutil.copytree(self.db_path, backup_path)
                logger.info(f"✅ Emergency backup created: {backup_path}")
                return backup_path
            else:
                logger.warning("⚠️ No database directory found to backup")
                return None

        except Exception as e:
            logger.error(f"❌ Failed to create emergency backup: {e}")
            return None

    def nuclear_reset_database(self) -> bool:
        """Completely remove and recreate the database directory"""
        try:
            logger.warning("🚨 NUCLEAR RESET: Completely removing database directory")

            if self.db_path.exists():
                # Remove lock files first
                for lock_file in self.db_path.glob("*.lock"):
                    try:
                        lock_file.unlink()
                        logger.info(f"🔓 Removed lock file: {lock_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove lock file {lock_file}: {e}")

                # Force remove even if files are locked
                if os.name == 'nt':  # Windows
                    os.system(f'rmdir /s /q "{self.db_path}"')
                else:  # Unix/Linux/Mac
                    shutil.rmtree(self.db_path, ignore_errors=True)

                # Double-check removal
                if self.db_path.exists():
                    logger.error("❌ Failed to remove database directory")
                    return False

            # Recreate empty directory
            self.db_path.mkdir(exist_ok=True)
            logger.info("✅ Database directory reset complete")
            return True

        except Exception as e:
            logger.error(f"❌ Nuclear reset failed: {e}")
            return False

    async def emergency_recovery(self) -> Dict[str, Any]:
        """Perform complete emergency recovery"""
        recovery_log = {
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "success": False,
            "backup_path": None,
            "extracted_documents": 0
        }

        try:
            # Step 1: Create emergency backup
            recovery_log["steps"].append("Creating emergency backup")
            backup_path = self.create_emergency_backup()
            if backup_path:
                recovery_log["backup_path"] = str(backup_path)

            # Step 2: Nuclear reset
            recovery_log["steps"].append("Performing nuclear database reset")
            reset_success = self.nuclear_reset_database()

            if reset_success:
                recovery_log["success"] = True
                recovery_log["steps"].append("Recovery completed successfully")
                logger.info("✅ Emergency recovery completed successfully")
            else:
                recovery_log["steps"].append("Nuclear reset failed")
                logger.error("❌ Emergency recovery failed")

        except Exception as e:
            recovery_log["steps"].append(f"Recovery failed with error: {e}")
            logger.error(f"❌ Emergency recovery failed: {e}")

        recovery_log["completed_at"] = datetime.now().isoformat()

        # Save recovery log
        try:
            log_file = self.backup_root / f"recovery_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w') as f:
                json.dump(recovery_log, f, indent=2)
            logger.info(f"📝 Recovery log saved: {log_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save recovery log: {e}")

        return recovery_log

async def perform_emergency_recovery() -> Dict[str, Any]:
    """Standalone function to perform emergency recovery"""
    recovery = EmergencyDBRecovery()
    return await recovery.emergency_recovery()

if __name__ == "__main__":
    import asyncio

    print("🚨 EMERGENCY ChromaDB Recovery")
    print("=" * 50)
    print("\nThis will:")
    print("1. Create emergency backup of current database")
    print("2. Extract any salvageable data")
    print("3. COMPLETELY RESET the database")
    print("4. You will lose all conversation history!")
    print("\nOnly use this if ChromaDB is completely broken!")

    response = input("\nAre you absolutely sure? Type 'RESET' to continue: ")
    if response == 'RESET':
        print("\n🚨 Starting emergency recovery...")
        result = asyncio.run(perform_emergency_recovery())

        print("\n📊 Recovery Results:")
        print(f"Success: {result['success']}")
        print(f"Documents extracted: {result['extracted_documents']}")
        if result['backup_path']:
            print(f"Backup saved: {result['backup_path']}")
        if result['extracted_data_path']:
            print(f"Data saved: {result['extracted_data_path']}")

        if result['success']:
            print("\n✅ Emergency recovery completed!")
            print("You can now restart Aura - it will create a fresh database.")
        else:
            print("\n❌ Emergency recovery failed!")
    else:
        print("❌ Emergency recovery cancelled")
