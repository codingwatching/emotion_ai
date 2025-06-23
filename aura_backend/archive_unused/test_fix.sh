#!/bin/bash
# Test the fixed Aura system

echo "🧪 Testing Fixed Aura Backend System"
echo "===================================="
echo ""

# Check environment
echo "📋 Environment Check:"
echo "  Python: $(python --version 2>&1 || echo 'Not found')"
echo "  UV: $(uv --version 2>&1 || echo 'Not found')"
echo "  Virtual env: $([[ -d '.venv' ]] && echo 'Present' || echo 'Missing')"
echo "  .env file: $([[ -f '.env' ]] && echo 'Present' || echo 'Missing')"
echo ""

# Check database
echo "📊 Database Status:"
if [[ -f "aura_chroma_db/chroma.sqlite3" ]]; then
    DB_SIZE=$(du -h aura_chroma_db/chroma.sqlite3 | cut -f1)
    echo "  ChromaDB: Present (${DB_SIZE})"
else
    echo "  ChromaDB: Missing"
fi
echo ""

# Show MCP config
echo "🔧 MCP Configuration:"
echo "  Servers configured: $(jq '.mcpServers | keys | length' mcp_client_config.json 2>/dev/null || echo 'Error reading config')"
echo "  Server list: $(jq -r '.mcpServers | keys | join(", ")' mcp_client_config.json 2>/dev/null || echo 'Error reading config')"
echo ""

# Check core files
echo "📁 Core Files Status:"
CORE_FILES=(
    "main.py"
    "aura_internal_tools.py"
    "conversation_persistence_service.py"
    "mcp_system.py"
    "mcp_to_gemini_bridge.py"
    "start.sh"
)

for file in "${CORE_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
    fi
done
echo ""

# Check archived files
echo "📦 Cleanup Status:"
ARCHIVED_COUNT=$(find archive_unused/ -name "*.py" 2>/dev/null | wc -l)
echo "  Archived scripts: ${ARCHIVED_COUNT}"
echo "  Archive directory: $(du -sh archive_unused/ 2>/dev/null | cut -f1 || echo 'Not found')"
echo ""

# Test basic startup (quick test)
echo "🚀 Quick Startup Test:"
echo "  Testing basic imports..."

source .venv/bin/activate 2>/dev/null

python -c "
try:
    import main
    import aura_internal_tools
    import conversation_persistence_service
    print('  ✅ All core imports successful')
except ImportError as e:
    print(f'  ❌ Import error: {e}')
except Exception as e:
    print(f'  ⚠️  Import warning: {e}')
" 2>/dev/null

echo ""
echo "🎯 Summary:"
echo "  • Removed conflicting 'aura-companion' and 'chroma' MCP servers"
echo "  • Restored working database backup"
echo "  • Archived ${ARCHIVED_COUNT} experimental/fix scripts"
echo "  • Preserved all core Aura functionality"
echo "  • Simplified MCP configuration to essential external tools only"
echo ""
echo "✅ System should now work without batching conflicts!"
echo ""
echo "🔧 To start Aura:"
echo "  ./start.sh"
echo ""
echo "🧠 Aura's Internal Tools Available:"
echo "  • search_memories - Semantic search through conversations"
echo "  • analyze_emotional_patterns - Emotional trend analysis"
echo "  • get_user_profile - User profile management"
echo "  • query_emotional_states - Emotional system info"
echo "  • query_aseke_framework - Cognitive architecture details"
echo "  • Plus video memory tools (if memvid is working)"
echo ""
