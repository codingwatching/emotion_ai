#!/usr/bin/env python3
"""
ChromaDB Recovery Tool - Rebuild Corrupted Database
==================================================

This tool rebuilds a corrupted ChromaDB database by:
1. Backing up the current database
2. Creating a fresh database
3. Optionally migrating data if possible
"""

import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBRecovery:
    def __init__(self, db_path: str = "./aura_chroma_db"):
        self.db_path = Path(db_path)
        self.backup_dir = self.db_path.parent / "chromadb_backups"
        self.backup_dir.mkdir(exist_ok=True)

    def backup_current_db(self) -> Path | None:
        """Create timestamped backup of current database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"

        if self.db_path.exists():
            logger.info(f"Creating backup at: {backup_path}")
            shutil.copytree(self.db_path, backup_path)
            return backup_path
        else:
            logger.warning("No existing database to backup")
            return None

    def extract_data_if_possible(self, backup_path: Path) -> dict:
        """Try to extract any salvageable data from the corrupted database"""
        extracted_data = {
            "conversations": [],
            "collections": [],
            "metadata": {}
        }

        try:
            # Connect to the backed up database
            db_file = backup_path / "chroma.sqlite3"
            if not db_file.exists():
                logger.warning("No SQLite file found in backup")
                return extracted_data

            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Try to get collection information
            try:
                cursor.execute("SELECT id, name, metadata FROM collections")
                collections = cursor.fetchall()
                extracted_data["collections"] = [
                    {"id": str(c[0]), "name": c[1], "metadata": c[2]}
                    for c in collections
                ]
                logger.info(f"Extracted {len(collections)} collections")
            except Exception as e:
                logger.error(f"Failed to extract collections: {e}")

            # Try to get embeddings data
            try:
                cursor.execute("""
                    SELECT e.id, e.collection_id, e.embedding, e.document, e.metadata
                    FROM embeddings e
                    LIMIT 1000
                """)
                embeddings = cursor.fetchall()
                logger.info(f"Extracted {len(embeddings)} embeddings")

                # Save a sample for analysis
                if embeddings:
                    extracted_data["sample_embeddings"] = len(embeddings)
            except Exception as e:
                logger.error(f"Failed to extract embeddings: {e}")

            conn.close()

        except Exception as e:
            logger.error(f"Failed to extract data: {e}")

        return extracted_data

    def clean_database(self):
        """Remove the corrupted database directory"""
        if self.db_path.exists():
            logger.info(f"Removing corrupted database at: {self.db_path}")
            shutil.rmtree(self.db_path)
            logger.info("Database directory removed")

    def create_recovery_report(self, backup_path: Path | None, extracted_data: dict):
        """Create a report of what was recovered"""
        report_path = self.backup_dir / "recovery_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "backup_path": str(backup_path) if backup_path else None,
            "original_path": str(self.db_path),
            "collections_found": len(extracted_data.get("collections", [])),
            "sample_embeddings_found": extracted_data.get("sample_embeddings", 0),
            "collections": extracted_data.get("collections", []),
            "status": "Recovery completed - database rebuilt",
            "next_steps": [
                "The database has been reset to a clean state",
                "Previous data has been backed up",
                "The system will recreate collections on next startup",
                "Monitor for any recurring errors"
            ]
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Recovery report saved to: {report_path}")
        return report

    def perform_recovery(self):
        """Main recovery process"""
        logger.info("=" * 60)
        logger.info("ChromaDB Recovery Process Started")
        logger.info("=" * 60)

        # Step 1: Backup
        backup_path = self.backup_current_db()

        # Step 2: Try to extract data
        extracted_data = {}
        if backup_path:
            extracted_data = self.extract_data_if_possible(backup_path)

        # Step 3: Clean the database
        self.clean_database()

        # Step 4: Create recovery report
        report = self.create_recovery_report(backup_path, extracted_data)

        logger.info("=" * 60)
        logger.info("Recovery Process Completed")
        logger.info("=" * 60)
        logger.info("\nSummary:")
        logger.info(f"- Backup created: {backup_path}")
        logger.info(f"- Collections found: {report['collections_found']}")
        logger.info("- Database cleaned and ready for fresh start")
        logger.info("\nThe system will recreate the database on next startup.")

        return report

def main():
    """Main recovery function with user confirmation"""
    print("\n🔧 ChromaDB Recovery Tool")
    print("========================")
    print("\nThis tool will:")
    print("1. Backup your current (possibly corrupted) database")
    print("2. Extract any recoverable information")
    print("3. Remove the corrupted database")
    print("4. Allow the system to create a fresh database on restart")
    print("\n⚠️  WARNING: This will reset your vector database!")
    print("All conversation history and patterns will be lost.")
    print("(A backup will be created first)")

    response = input("\nDo you want to proceed? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        recovery = ChromaDBRecovery()
        recovery.perform_recovery()

        print("\n✅ Recovery completed!")
        print("\nNext steps:")
        print("1. The system will auto-restart and create a fresh database")
        print("2. Monitor the logs for any new errors")
        print("3. The backup is saved in ./chromadb_backups/")
    else:
        print("\n❌ Recovery cancelled")

if __name__ == "__main__":
    main()
