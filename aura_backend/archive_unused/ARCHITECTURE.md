# Aura Architecture

## Components

### 1. Aura Core API (`main.py`)
- REST API server on port 8000
- Handles conversations, memory, emotional analysis
- Integrates with MCP client for external tools

### 2. Aura Internal Server (`aura_server.py`)
- Exposes Aura's capabilities as MCP tools
- Runs as a FastMCP server
- Provides: memory search, emotional analysis, user profiles, etc.

### 3. MCP Client Integration (`mcp_integration.py`)
- Connects to external MCP servers (sqlite, docker, etc.)
- Also connects to Aura's internal server
- Provides unified tool interface to the AI

## Service Dependencies

```
Frontend (port 5173)
    ↓
Aura Core API (port 8000)
    ↓
    ├── Vector DB (ChromaDB)
    ├── File System (./aura_data)
    └── MCP Client
         ├── Aura Internal Server (stdio)
         └── External MCP Servers (stdio)
```

## Startup Order

1. Start Aura Internal Server (background)
2. Start Core API (which initializes MCP client)
3. Start Frontend (optional, for UI access)
