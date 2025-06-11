#!/usr/bin/env python3
"""
ChromaDB Recovery and Maintenance Script for Aura
================================================

This script provides utilities to diagnose and fix ChromaDB issues,
particularly the "Failed to apply logs to the metadata segment" error.
"""

import sqlite3
import shutil
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChromaDBRecovery:
    """Handles recovery and maintenance of ChromaDB databases"""

    def __init__(self, db_path: str = "./aura_chroma_db"):
        self.db_path = Path(db_path)
        self.sqlite_path = self.db_path / "chroma.sqlite3"
        self.backup_dir = self.db_path.parent / "chromadb_backups"

    def backup_database(self) -> Path:
        """Create a backup of the current database"""
        self.backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"chroma_backup_{timestamp}"

        logger.info(f"Creating backup at: {backup_path}")
        shutil.copytree(self.db_path, backup_path)
        return backup_path

    def check_database_integrity(self) -> bool:
        """Check SQLite database integrity"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchall()

            conn.close()

            if result[0][0] == "ok":
                logger.info("✅ Database integrity check passed")
                return True
            else:
                logger.error(f"❌ Database integrity check failed: {result}")
                return False

        except Exception as e:
            logger.error(f"❌ Error checking database integrity: {e}")
            return False

    def clean_wal_files(self):
        """Clean up WAL (Write-Ahead Log) files"""
        wal_file = self.db_path / "chroma.sqlite3-wal"
        shm_file = self.db_path / "chroma.sqlite3-shm"

        if wal_file.exists():
            logger.info(f"Removing WAL file: {wal_file}")
            wal_file.unlink()

        if shm_file.exists():
            logger.info(f"Removing SHM file: {shm_file}")
            shm_file.unlink()

    def vacuum_database(self):
        """Vacuum the SQLite database to optimize and clean it"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            logger.info("Running VACUUM on database...")
            cursor.execute("VACUUM")
            conn.commit()

            logger.info("Running REINDEX...")
            cursor.execute("REINDEX")
            conn.commit()

            conn.close()
            logger.info("✅ Database optimization completed")

        except Exception as e:
            logger.error(f"❌ Error during database optimization: {e}")
            raise

    def fix_permissions(self):
        """Fix file permissions to be more secure"""
        # Set appropriate permissions (read/write for owner, read for group)
        os.chmod(self.db_path, 0o755)

        for file in self.db_path.glob("*"):
            if file.is_file():
                os.chmod(file, 0o644)
            elif file.is_dir():
                os.chmod(file, 0o755)

        logger.info("✅ Fixed file permissions")

    def check_embeddings_queue(self):
        """Check for duplicate entries in embeddings_queue table"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            # Check if embeddings_queue table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='embeddings_queue'
            """)

            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM embeddings_queue")
                count = cursor.fetchone()[0]
                logger.info(f"Embeddings queue contains {count} entries")

                if count > 0:
                    logger.warning("⚠️ Found entries in embeddings_queue - this might cause issues")

            conn.close()

        except Exception as e:
            logger.error(f"❌ Error checking embeddings queue: {e}")

    def clear_embeddings_queue(self):
        """Clear the embeddings queue to resolve stuck processing"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            # Check if embeddings_queue table exists and has entries
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='embeddings_queue'
            """)

            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM embeddings_queue")
                count = cursor.fetchone()[0]

                if count > 0:
                    logger.warning(f"⚠️ Found {count} stuck entries in embeddings_queue")
                    logger.info("🧹 Clearing embeddings queue...")

                    cursor.execute("DELETE FROM embeddings_queue")
                    conn.commit()

                    logger.info(f"✅ Cleared {count} entries from embeddings_queue")
                else:
                    logger.info("✅ Embeddings queue is already empty")
            else:
                logger.info("ℹ️ No embeddings_queue table found")

            conn.close()

        except Exception as e:
            logger.error(f"❌ Error clearing embeddings queue: {e}")
            raise

    def force_wal_checkpoint(self):
        """Force WAL checkpoint to resolve compaction issues"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            logger.info("🔄 Forcing WAL checkpoint...")

            # Force checkpoint to merge WAL into main database
            cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            result = cursor.fetchone()
            logger.info(f"WAL checkpoint result: {result}")

            # Disable WAL mode temporarily and re-enable to force cleanup
            cursor.execute("PRAGMA journal_mode=DELETE")
            cursor.execute("PRAGMA journal_mode=WAL")

            conn.commit()
            conn.close()

            logger.info("✅ WAL checkpoint completed")

        except Exception as e:
            logger.error(f"❌ Error during WAL checkpoint: {e}")
            # Don't raise - this might fail in some cases but still help

    def analyze_collections(self):
        """Analyze collection state and identify issues"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            logger.info("📊 Analyzing ChromaDB collections...")

            # Check collections table
            cursor.execute("""
                SELECT id, name, dimension FROM collections
            """)
            collections = cursor.fetchall()

            logger.info(f"Found {len(collections)} collections:")
            for collection_id, name, dimension in collections:
                logger.info(f"  - {name} (ID: {collection_id}, Dimension: {dimension})")

                # Check embeddings count for each collection
                cursor.execute("""
                    SELECT COUNT(*) FROM embeddings WHERE collection_id = ?
                """, (collection_id,))
                embedding_count = cursor.fetchone()[0]

                logger.info(f"    └─ {embedding_count} embeddings")

                # Check for orphaned segments
                cursor.execute("""
                    SELECT COUNT(*) FROM segments WHERE collection_id = ?
                """, (collection_id,))
                segment_count = cursor.fetchone()[0]

                logger.info(f"    └─ {segment_count} segments")

            conn.close()

        except Exception as e:
            logger.error(f"❌ Error analyzing collections: {e}")

    def rebuild_search_index(self):
        """Attempt to rebuild search indices"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            logger.info("🔨 Rebuilding search indices...")

            # Get all table names
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table'
            """)
            tables = cursor.fetchall()

            for (table_name,) in tables:
                try:
                    logger.info(f"Reindexing table: {table_name}")
                    cursor.execute(f"REINDEX {table_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not reindex {table_name}: {e}")

            conn.commit()
            conn.close()

            logger.info("✅ Index rebuild completed")

        except Exception as e:
            logger.error(f"❌ Error rebuilding indices: {e}")

    def check_compaction_state(self):
        """Check for signs of failed compaction"""
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()

            logger.info("🔍 Checking compaction state...")

            # Check for temporary tables that might indicate stuck compaction
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE '%temp%'
            """)
            temp_tables = cursor.fetchall()

            if temp_tables:
                logger.warning(f"⚠️ Found {len(temp_tables)} temporary tables - possible stuck compaction")
                for (table_name,) in temp_tables:
                    logger.warning(f"  - {table_name}")

            # Check segments table for inconsistencies
            cursor.execute("""
                SELECT collection_id, COUNT(*) as segment_count
                FROM segments
                GROUP BY collection_id
            """)
            segment_counts = cursor.fetchall()

            for collection_id, count in segment_counts:
                if count > 10:  # Unusually high number of segments
                    logger.warning(f"⚠️ Collection {collection_id} has {count} segments (may need compaction)")

            conn.close()

        except Exception as e:
            logger.error(f"❌ Error checking compaction state: {e}")

    def emergency_recovery(self):
        """Emergency recovery for severely corrupted databases"""
        logger.info("🚨 Starting EMERGENCY recovery process...")

        # Step 1: Backup current database
        backup_path = self.backup_database()
        logger.info(f"✅ Emergency backup created at: {backup_path}")

        # Step 2: Force WAL checkpoint before anything else
        self.force_wal_checkpoint()

        # Step 3: Clear stuck embeddings queue
        self.clear_embeddings_queue()

        # Step 4: Analyze current state
        self.analyze_collections()
        self.check_compaction_state()

        # Step 5: Clean WAL files
        self.clean_wal_files()

        # Step 6: Rebuild indices
        self.rebuild_search_index()

        # Step 7: Final vacuum
        try:
            self.vacuum_database()
        except Exception as e:
            logger.error(f"Failed final vacuum: {e}")

        # Step 8: Fix permissions
        self.fix_permissions()

        logger.info("✅ Emergency recovery completed")

    def recover_database(self):
        """Perform full database recovery"""
        logger.info("🔧 Starting ChromaDB recovery process...")

        # Step 1: Backup current database
        backup_path = self.backup_database()
        logger.info(f"✅ Backup created at: {backup_path}")

        # Step 2: Check integrity
        if not self.check_database_integrity():
            logger.warning("⚠️ Database integrity check failed - attempting recovery")

        # Step 3: Force WAL checkpoint
        self.force_wal_checkpoint()

        # Step 4: Clear embeddings queue if stuck
        self.clear_embeddings_queue()

        # Step 5: Clean WAL files
        self.clean_wal_files()

        # Step 6: Vacuum and optimize
        try:
            self.vacuum_database()
        except Exception as e:
            logger.error(f"Failed to vacuum: {e}")

        # Step 7: Fix permissions
        self.fix_permissions()

        # Step 8: Analyze final state
        self.analyze_collections()

        logger.info("✅ Recovery process completed")

def main():
    """Main recovery function"""
    recovery = ChromaDBRecovery("/home/ty/Repositories/ai_workspace/emotion_ai/aura_backend/aura_chroma_db")

    print("🔧 ChromaDB Recovery Tool for Aura")
    print("==================================")
    print("\nRecovery options:")
    print("1. Standard recovery (recommended)")
    print("2. Emergency recovery (for severe corruption)")
    print("3. Just clear embeddings queue")
    print("4. Just analyze database state")
    print()

    choice = input("Select option (1-4): ")

    if choice == "1":
        print("\n🔧 Running standard recovery...")
        recovery.recover_database()
        print("\n✅ Standard recovery completed! Restart Aura to test.")

    elif choice == "2":
        print("\n🚨 Running emergency recovery...")
        print("This will aggressively fix database issues.")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() == 'y':
            recovery.emergency_recovery()
            print("\n✅ Emergency recovery completed! Restart Aura to test.")
        else:
            print("❌ Emergency recovery cancelled")

    elif choice == "3":
        print("\n🧹 Clearing embeddings queue...")
        recovery.clear_embeddings_queue()
        print("\n✅ Embeddings queue cleared! Restart Aura to test.")

    elif choice == "4":
        print("\n📊 Analyzing database state...")
        recovery.analyze_collections()
        recovery.check_compaction_state()
        print("\n✅ Analysis completed!")

    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    main()
