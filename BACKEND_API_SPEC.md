# Backend API Endpoints Required

Based on the frontend code, here are the backend endpoints that need to be implemented:

## 1. Health Check
- **Endpoint**: `GET /health`
- **Purpose**: Check if backend is running
- **Expected Response**: JSON object with status information

## 2. Chat/Conversation
- **Endpoint**: `POST /conversation`
- **Purpose**: Send user message and get AI response
- **Request Body**:
```json
{
  "user_id": "string",
  "message": "string",
  "session_id": "string (optional)"
}
```
- **Expected Response**:
```json
{
  "response": "string",
  "emotional_state": {
    "name": "string",
    "intensity": "string",
    "brainwave": "string",
    "neurotransmitter": "string"
  },
  "cognitive_state": {
    "focus": "string",
    "description": "string"
  },
  "session_id": "string",
  "thinking_time": 1500,
  "model_info": {
    "name": "string",
    "supports_thinking": true
  }
}
```

## 3. Memory Search
- **Endpoint**: `POST /search`
- **Purpose**: Search through conversation memory
- **Request Body**:
```json
{
  "user_id": "string",
  "query": "string",
  "n_results": 10
}
```
- **Expected Response**:
```json
{
  "results": [
    {
      "content": "string",
      "metadata": {},
      "similarity": 0.95
    }
  ]
}
```

## 4. Chat History
- **Endpoint**: `GET /chat-history/{user_id}?limit=20`
- **Purpose**: Get user's conversation history
- **Expected Response**:
```json
{
  "sessions": [
    {
      "session_id": "string",
      "timestamp": "ISO date string",
      "last_message": "string",
      "message_count": 5
    }
  ]
}
```

## 5. Load Specific Chat Session
- **Endpoint**: `GET /chat-history/{user_id}/{session_id}`
- **Purpose**: Get messages from a specific session
- **Expected Response**:
```json
[
  {
    "content": "string",
    "sender": "user" | "aura"
  }
]
```

## 6. Emotional Analysis
- **Endpoint**: `GET /emotional-analysis/{user_id}?period={period}`
- **Purpose**: Get emotional patterns and insights
- **Expected Response**:
```json
{
  "dominant_emotion": "string",
  "emotional_stability": 0.75,
  "total_patterns": 50,
  "emotional_patterns": [
    {
      "emotion": "Happy",
      "count": 15,
      "percentage": 30.0
    }
  ],
  "recommendations": ["string"]
}
```

## 7. Video Archive Status (MemVid)
- **Endpoint**: `GET /memvid/status`
- **Purpose**: Get video memory archive status
- **Expected Response**:
```json
{
  "status": "operational",
  "archives_count": 4,
  "archives": ["archive1", "archive2"]
}
```

All endpoints should return appropriate HTTP status codes and error messages when things go wrong.

## Database Recovery Notes

If you're seeing ChromaDB search failures like in the terminal:
- `ERROR:conversation_persistence_service: search_conversations failed with non-recoverable database error`
- `CRITICAL:conversation_persistence_service: Emergency recovery trigger failed`

This suggests the ChromaDB database may be corrupted. Try:
1. Stop the backend server
2. Delete/backup the ChromaDB data directory
3. Restart the backend to recreate the database
4. Or run any database recovery scripts you have

8000
{
  "message": "Aura Backend - Advanced AI Companion",
  "status": "operational",
  "features": [
    "Vector Database Integration",
    "MCP Server Support",
    "Advanced State Management",
    "Emotional Pattern Analysis",
    "Cognitive Focus Tracking"
  ]
}

8000/docs

Aura Backend
 1.0.0
OAS 3.1
/openapi.json
Advanced AI Companion with Vector Database and MCP Integration

MCP Integration


GET
/mcp/tools
List Mcp Tools


GET
/mcp/prompts
List Mcp Prompts


POST
/mcp/tools/execute
Execute Tool Endpoint


POST
/mcp/prompts/get
Get Prompt Endpoint


GET
/mcp/status
Mcp Status

default


GET
/
Root


GET
/health
Wrapper


POST
/conversation
Process Conversation


POST
/search
Search Memories


GET
/emotional-analysis/{user_id}
Get Emotional Analysis


POST
/export/{user_id}
Export User Data


GET
/chat-history/{user_id}
Get Chat History


GET
/chat-history/{user_id}/{session_id}
Get Session Messages


DELETE
/chat-history/{user_id}/{session_id}
Delete Chat Session


POST
/mcp/execute-tool
Mcp Execute Tool


DELETE
/sessions/{user_id}
Clear User Sessions


GET
/mcp/bridge-status
Get Mcp Bridge Status


GET
/mcp/system-status
Get Mcp System Status


GET
/persistence/health
Get Persistence Health


POST
/test/persistence
Test Persistence Reliability


GET
/memvid/status
Get Memvid Status


GET
/vector-db/health
Get Vector Db Health


GET
/database-protection/status
Get Database Protection Status


POST
/database-protection/emergency-backup
Trigger Emergency Backup


GET
/autonomic/status
Get Autonomic Status


GET
/autonomic/tasks/{user_id}
Get User Autonomic Tasks


GET
/autonomic/task/{task_id}
Get Autonomic Task Details


POST
/autonomic/submit-task
Submit Autonomic Task


GET
/autonomic/task/{task_id}/result
Get Autonomic Task Result


POST
/autonomic/control/{action}
Control Autonomic System


POST
/vector-db/optimize
Optimize Vector Db


Schemas
ConversationRequestExpand allobject
ConversationResponseExpand allobject
ExecuteToolRequestExpand allobject
HTTPValidationErrorExpand allobject
MCPPromptRequestExpand allobject
MCPToolRequestExpand allobject
SearchRequestExpand allobject
ValidationErrorExpand allobject