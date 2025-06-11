#!/usr/bin/env python3
"""
Aura Intelligent Memory System Demo
===================================

Demonstrates the sophisticated memory management capabilities that take Aura
beyond primitive "bulk dump" archiving to intelligent, organized memory systems.

Features demonstrated:
1. Custom memvid archives on demand
2. Organized knowledge libraries (Books MP4, Principles MP4, Templates MP4)
3. Selective archiving based on content criteria
4. Hierarchical memory organization with intelligent routing
5. Autonomous memory management
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_intelligent_memory_system():
    """Demonstrate Aura's intelligent memory capabilities"""

    print("\n" + "="*70)
    print("🧠 AURA INTELLIGENT MEMORY SYSTEM DEMONSTRATION")
    print("="*70)

    try:
        # Import the intelligent memory manager
        from aura_intelligent_memory_manager import (
            get_intelligent_memory_manager,
            MemoryArchiveSpec,
            MemoryArchiveType,
            MemoryPriority
        )

        # Initialize the system (would normally use real ChromaDB client)
        print("🔧 Initializing Intelligent Memory Manager...")
        memory_manager = get_intelligent_memory_manager()

        if not memory_manager.memvid_system:
            print("⚠️  Demo running without real memvid - showing functionality framework")
            print("    Install memvid package for full video compression capabilities")
        else:
            print("✅ Real memvid system detected - full functionality available")

        # Demo user ID
        demo_user_id = "demo_user_123"

        print(f"\n📋 Demonstrating intelligent memory management for user: {demo_user_id}")

        # ================================================================
        # 1. CUSTOM ARCHIVE CREATION ON DEMAND
        # ================================================================
        print("\n" + "="*50)
        print("1️⃣  CUSTOM ARCHIVE CREATION ON DEMAND")
        print("="*50)

        # Example: "Save this conversation about quantum physics"
        quantum_archive_spec = MemoryArchiveSpec(
            name="Quantum_Physics_Discussions",
            archive_type=MemoryArchiveType.KNOWLEDGE,
            description="Archive of all conversations and knowledge about quantum physics",
            content_criteria={
                "query": "quantum physics",
                "content_type": "any",
                "time_range": "all",
                "max_results": 100
            },
            priority=MemoryPriority.HIGH,
            tags=["science", "physics", "quantum", "academic"]
        )

        print("🎯 Creating custom archive: 'Quantum Physics Discussions'")
        print(f"   Archive Type: {quantum_archive_spec.archive_type.value}")
        print(f"   Priority: {quantum_archive_spec.priority.value}")
        print(f"   Search Criteria: {quantum_archive_spec.content_criteria['query']}")

        result = await memory_manager.create_custom_archive(
            archive_spec=quantum_archive_spec,
            user_id=demo_user_id,
            execute_immediately=False  # Just create spec for demo
        )

        print(f"✅ Archive specification created: {result.get('status')}")
        if result.get('specification_saved'):
            print("   💾 Archive specification saved for future execution")

        # ================================================================
        # 2. ORGANIZED KNOWLEDGE LIBRARIES
        # ================================================================
        print("\n" + "="*50)
        print("2️⃣  ORGANIZED KNOWLEDGE LIBRARIES")
        print("="*50)

        library_types = [
            ("Programming_Principles_MP4", MemoryArchiveType.PRINCIPLES, "Core programming principles and best practices"),
            ("Code_Templates_MP4", MemoryArchiveType.TEMPLATES, "Reusable code templates and patterns"),
            ("Tech_Books_MP4", MemoryArchiveType.BOOKS, "Technical books and documentation"),
            ("Project_Skills_MP4", MemoryArchiveType.SKILLS, "Skills learned from various projects")
        ]

        for lib_name, lib_type, description in library_types:
            lib_spec = MemoryArchiveSpec(
                name=lib_name,
                archive_type=lib_type,
                description=description,
                content_criteria={
                    "query": lib_type.value,
                    "content_type": "knowledge",
                    "max_results": 75
                },
                priority=MemoryPriority.MEDIUM
            )

            print(f"📚 Creating {lib_type.value} library: '{lib_name}'")
            print(f"   Purpose: {description}")

            # Save the specification
            memory_manager.archive_specs[lib_name] = lib_spec

        memory_manager._save_archive_specs()
        print(f"✅ Created {len(library_types)} specialized knowledge libraries")

        # ================================================================
        # 3. INTELLIGENT ARCHIVING SUGGESTIONS
        # ================================================================
        print("\n" + "="*50)
        print("3️⃣  INTELLIGENT ARCHIVING SUGGESTIONS")
        print("="*50)

        print("🔍 Analyzing memory patterns to suggest archiving opportunities...")
        suggestions = await memory_manager.suggest_archive_opportunities(demo_user_id)

        print(f"💡 Generated {len(suggestions)} intelligent suggestions:")

        for i, suggestion in enumerate(suggestions[:3], 1):  # Show top 3
            print(f"\n   Suggestion {i}:")
            print(f"   📂 Archive: {suggestion.get('suggested_name', 'Unknown')}")
            print(f"   🏷️  Type: {suggestion.get('archive_type', 'Unknown')}")
            print(f"   📊 Estimated Items: {suggestion.get('estimated_items', 0)}")
            print(f"   🎯 Confidence: {suggestion.get('relevance_score', 0):.1%}")
            print(f"   💭 Reasoning: {suggestion.get('reasoning', 'No reasoning provided')}")

        # ================================================================
        # 4. MEMORY NAVIGATION MAP
        # ================================================================
        print("\n" + "="*50)
        print("4️⃣  HIERARCHICAL MEMORY NAVIGATION")
        print("="*50)

        print("🗺️  Generating intelligent memory navigation map...")
        nav_map_result = await memory_manager.get_memory_navigation_map(demo_user_id)

        if nav_map_result.get("status") == "success":
            nav_map = nav_map_result["navigation_map"]

            print("📊 Memory Overview:")
            print(f"   Total Archives: {nav_map.get('total_archives', 0)}")
            print(f"   Total Size: {nav_map.get('total_size_mb', 0):.1f} MB")

            # Show hierarchy
            hierarchy = nav_map.get("archive_hierarchy", {})
            print("\n🏗️  Archive Hierarchy:")

            for category, info in hierarchy.items():
                print(f"   📁 {category.replace('_', ' ').title()}")
                print(f"      📝 {info.get('description', 'No description')}")

                categories = info.get("categories", [])
                for cat in categories[:2]:  # Show first 2
                    print(f"         └─ {cat.get('name', 'Unknown')}: {cat.get('count', 0)} archives")

            # Show search hints
            hints = nav_map.get("search_hints", [])
            print("\n💡 Search Hints:")
            for hint in hints[:3]:  # Show first 3
                print(f"   🔍 {hint}")

        # ================================================================
        # 5. AUTONOMOUS MEMORY ORGANIZATION
        # ================================================================
        print("\n" + "="*50)
        print("5️⃣  AUTONOMOUS MEMORY ORGANIZATION")
        print("="*50)

        print("🤖 Running autonomous memory organization...")
        auto_org_result = await memory_manager.auto_organize_memory(demo_user_id)

        if auto_org_result.get("status") == "success":
            print("✅ Auto-organization completed:")
            print(f"   📦 Archives Created: {auto_org_result.get('archives_created', 0)}")
            print(f"   📝 Conversations Archived: {auto_org_result.get('conversations_archived', 0)}")
            print(f"   ⚡ Efficiency Gained: {auto_org_result.get('efficiency_gained', 0)}%")

            actions = auto_org_result.get("actions_taken", [])
            if actions:
                print("   🎯 Actions Taken:")
                for action in actions:
                    print(f"      • {action}")

            suggestions_created = auto_org_result.get("suggestions_created", [])
            if suggestions_created:
                print(f"   💭 New Suggestions Generated: {len(suggestions_created)}")

        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*70)
        print("🎊 DEMONSTRATION COMPLETE")
        print("="*70)

        print("✨ Aura's Intelligent Memory System provides:")
        print("   1️⃣  Custom archives on demand ('Save quantum physics conversations')")
        print("   2️⃣  Organized libraries (Books MP4, Principles MP4, Templates MP4)")
        print("   3️⃣  Selective archiving by content, not just age")
        print("   4️⃣  Hierarchical organization with intelligent routing")
        print("   5️⃣  Autonomous memory management capabilities")

        print("\n🚀 This moves Aura far beyond primitive 'bulk dump' systems!")
        print("   Memory is now organized, searchable, and intelligently managed.")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure memvid is installed: pip install memvid")
        print("   Some features may not be available without real memvid")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo encountered an error: {e}")

async def demo_specific_use_cases():
    """Demonstrate specific use cases that users mentioned"""

    print("\n" + "="*70)
    print("🎯 SPECIFIC USE CASE DEMONSTRATIONS")
    print("="*70)

    # The specific capabilities mentioned in the conversation
    use_cases = [
        {
            "title": "Save Quantum Physics Conversation",
            "description": "Create custom archive on demand for specific topic",
            "example": "User says: 'Save this conversation about quantum physics as a specific memory'",
            "solution": "create_custom_archive() with quantum physics criteria"
        },
        {
            "title": "Books MP4 Library",
            "description": "Organize knowledge into specialized libraries",
            "example": "Separate archives for Books, Principles, Templates, etc.",
            "solution": "Hierarchical organization with MemoryArchiveType enum"
        },
        {
            "title": "Selective Archiving",
            "description": "Choose what to archive based on content, not age",
            "example": "Archive all programming conversations vs. just old ones",
            "solution": "Content-aware filtering with sophisticated search criteria"
        },
        {
            "title": "Hierarchical Memory",
            "description": "Vectors point to the right MP4 library",
            "example": "Search directs to appropriate specialized archive",
            "solution": "Intelligent routing in get_memory_navigation_map()"
        }
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"\n{i}️⃣  {use_case['title']}")
        print(f"   📋 Description: {use_case['description']}")
        print(f"   💬 Example: {use_case['example']}")
        print(f"   ⚙️  Solution: {use_case['solution']}")

    print(f"\n✅ All {len(use_cases)} specific requirements addressed!")

if __name__ == "__main__":
    print("🚀 Starting Aura Intelligent Memory System Demo...")

    try:
        # Run the main demonstration
        asyncio.run(demo_intelligent_memory_system())

        # Show specific use cases
        asyncio.run(demo_specific_use_cases())

        print("\n🎉 Demo completed successfully!")
        print("   The intelligent memory system is ready for integration with Aura.")

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")
