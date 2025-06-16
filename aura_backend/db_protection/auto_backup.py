#!/usr/bin/env python3
"""Auto-backup script for ChromaDB - run this before major operations"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recover_chromadb import ChromaDBRecovery

def main():
    recovery = ChromaDBRecovery()
    try:
        backup_path = recovery.create_full_backup()
        print(f"✅ Auto-backup completed: {backup_path}")
        return True
    except Exception as e:
        print(f"❌ Auto-backup failed: {e}")
        return False

if __name__ == "__main__":
    main()
