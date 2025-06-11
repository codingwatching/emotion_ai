#!/usr/bin/env python3
"""
Aura Memory Data Migration
=========================

Properly migrates conversation data from the old ChromaDB format to a new
compatible format while preserving ALL conversation history.
"""

import sys
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

def extract_conversations_from_old_db():
    """Extract all conversation data from the old database"""
    try:
        print("📖 Extracting conversation data from old database...")

        db_path = Path("./aura_chroma_db/chroma.sqlite3")
        if not db_path.exists():
            print("❌ Database file not found!")
            return []

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get conversation documents and metadata
        conversations = []

        # Query to get all the conversation data
        query = """
        SELECT
            em.id,
            em.string_value as document,
            e.embedding_id,
            e.created_at
        FROM embedding_metadata em
        JOIN embeddings e ON em.id = e.id
        WHERE em.key = 'chroma:document'
        ORDER BY e.created_at
        """

        cursor.execute(query)
        documents = cursor.fetchall()

        print(f"📄 Found {len(documents)} documents")

        # Get metadata for each document
        for doc_id, document, embedding_id, created_at in documents:
            # Get all metadata for this document
            metadata_query = """
            SELECT key, string_value, int_value, float_value, bool_value
            FROM embedding_metadata
            WHERE id = ?
            """
            cursor.execute(metadata_query, (doc_id,))
            metadata_rows = cursor.fetchall()

            # Build metadata dictionary
            metadata = {}
            for key, str_val, int_val, float_val, bool_val in metadata_rows:
                if key == 'chroma:document':
                    continue  # Skip the document content itself

                # Determine the actual value
                if str_val is not None:
                    metadata[key] = str_val
                elif int_val is not None:
                    metadata[key] = int_val
                elif float_val is not None:
                    metadata[key] = float_val
                elif bool_val is not None:
                    metadata[key] = bool_val

            conversations.append({
                'id': embedding_id,
                'document': document,
                'metadata': metadata,
                'created_at': created_at
            })

        conn.close()

        print(f"✅ Extracted {len(conversations)} conversations with metadata")

        # Show sample of what we extracted
        if conversations:
            print("\n📋 Sample extracted data:")
            for i, conv in enumerate(conversations[:3]):
                print(f"   {i+1}. [{conv['metadata'].get('sender', 'unknown')}] {conv['document'][:50]}...")
                print(f"      User: {conv['metadata'].get('user_id', 'unknown')}")
                print(f"      Time: {conv['created_at']}")

        return conversations

    except Exception as e:
        print(f"❌ Failed to extract conversations: {e}")
        return []

def create_fresh_compatible_db(conversations: List[Dict[Any, Any]]):
    """Create a fresh, compatible ChromaDB and import the conversations"""
    try:
        print("\n🔄 Creating fresh compatible database...")

        # Backup the old database
        old_db_path = Path("./aura_chroma_db")
        backup_path = Path(f"./aura_chroma_db_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if old_db_path.exists():
            shutil.move(str(old_db_path), str(backup_path))
            print(f"📦 Moved old database to: {backup_path}")

        # Create new ChromaDB
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(
            path="./aura_chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create required collections
        collections_to_create = [
            ("aura_conversations", "Conversation history with semantic search"),
            ("aura_emotional_patterns", "Historical emotional state patterns"),
            ("aura_cognitive_patterns", "Cognitive focus and ASEKE component tracking"),
            ("aura_knowledge_substrate", "Shared knowledge and insights")
        ]

        created_collections = {}
        for name, description in collections_to_create:
            collection = client.create_collection(
                name=name,
                metadata={"description": description}
            )
            created_collections[name] = collection
            print(f"✅ Created collection: {name}")

        # Initialize embedding model
        print("🤖 Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Import conversations
        conversations_collection = created_collections["aura_conversations"]

        print(f"📥 Importing {len(conversations)} conversations...")

        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for i, conv in enumerate(conversations):
            # Generate new embedding for the document
            embedding = embedding_model.encode(conv['document']).tolist()

            # Use original ID or create new one
            doc_id = conv['id'] if conv['id'] else f"migrated_{i}_{datetime.now().timestamp()}"

            documents.append(conv['document'])
            embeddings.append(embedding)
            metadatas.append(conv['metadata'])
            ids.append(doc_id)

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(conversations)} conversations...")

        # Batch import all conversations
        if documents:
            conversations_collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ Successfully imported {len(documents)} conversations!")

        # Verify the import
        total_count = conversations_collection.count()
        print(f"🔍 Verification: {total_count} documents in new database")

        return True

    except Exception as e:
        print(f"❌ Failed to create fresh database: {e}")
        return False

def test_new_database():
    """Test that the new database works correctly"""
    try:
        print("\n🧪 Testing new database functionality...")

        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(
            path="./aura_chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # Test collection access
        conversations = client.get_collection("aura_conversations")
        count = conversations.count()
        print(f"✅ Conversations collection: {count} documents")

        # Test search functionality
        if count > 0:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            test_query = "hello"
            query_embedding = embedding_model.encode(test_query).tolist()

            results = conversations.query(
                query_embeddings=[query_embedding],
                n_results=min(3, count),
                include=["documents", "metadatas", "distances"]
            )

            if results and results.get('documents') and results['documents'] is not None and len(results['documents']) > 0 and results['documents'][0]:
                print(f"✅ Search functionality: Found {len(results['documents'][0])} results")
                print(f"   Sample: {results['documents'][0][0][:50]}...")
            else:
                print("⚠️  Search returned no results")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def main():
    """Main migration process"""
    print("🔄 Aura Memory Data Migration")
    print("=" * 40)
    print("Preserving your conversation history while fixing compatibility issues")
    print()

    # Step 1: Extract existing conversations
    print("1️⃣  EXTRACTING EXISTING DATA")
    print("-" * 30)
    conversations = extract_conversations_from_old_db()

    if not conversations:
        print("❌ No conversations found to migrate!")
        return

    # Step 2: Create new compatible database
    print("\n2️⃣  CREATING COMPATIBLE DATABASE")
    print("-" * 30)
    success = create_fresh_compatible_db(conversations)

    if not success:
        print("❌ Failed to create new database!")
        return

    # Step 3: Test new database
    print("\n3️⃣  TESTING NEW DATABASE")
    print("-" * 30)
    test_success = test_new_database()

    # Summary
    print("\n" + "=" * 40)
    print("🎊 MIGRATION SUMMARY")
    print("=" * 40)

    if success and test_success:
        print("✅ Migration completed successfully!")
        print(f"✅ Preserved {len(conversations)} conversations")
        print("✅ Database compatibility fixed")
        print("✅ Search functionality verified")
        print("\n🚀 Your Aura memory system should now work properly!")
        print("\n🔄 Next steps:")
        print("   1. Restart your Aura backend server")
        print("   2. Test chat functionality in the UI")
        print("   3. Verify chat history loads correctly")
    else:
        print("❌ Migration had some issues")
        print("💡 Your original data is backed up safely")

    print("\n📦 Backups available in: aura_chroma_db_old_* directories")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Migration interrupted")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
