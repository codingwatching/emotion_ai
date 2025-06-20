# Aura Backend - Advanced AI Companion

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Vector DB](https://img.shields.io/badge/ChromaDB-latest-purple.svg)](https://chromadb.ai)
[![MCP](https://img.shields.io/badge/MCP-enabled-orange.svg)](https://modelcontextprotocol.io)

> **Sophisticated AI Companion with Vector Database, Emotional Intelligence, and Model Context Protocol Integration**

# Two WARNINGS and a disclaimer

- AI generated code

- Aura could be dangerous despite my attempted safeguards in a number of ways including but not limited to
 PC damage
 User mental health and attachment
 Emotional agentic activity

# User assumes all liability.

![alt text](image-4.png)

Chat storage currently broken sorry, but memory working well- not a coder 6/20/25

![alt text](image-1.png)

## 🌟 Features

### 🧠 Advanced Cognitive Architecture

- **ASEKE Framework**: Adaptive Socio-Emotional Knowledge Ecosystem
- **Real-time Emotional State Detection** with neurological correlations
- **Cognitive Focus Tracking** across different mental frameworks
- **Adaptive Self-Reflection** for continuous improvement
- **🆕 Thinking Extraction**: Transparent AI reasoning with thought analysis and cognitive transparency

### 🗄️ Intelligent Memory System

- **Vector Database Integration** with ChromaDB for semantic search
- **Persistent Conversation Memory** with embedding-based retrieval
- **Emotional Pattern Analysis** over time
- **Cognitive State Tracking** and trend analysis
- **MemVid AI QR code mp4 memory** Infinite MP4 based memory
- **Internal AI guided Memory Organization tools** Move information from short to long term memory systems to avoid bottlenecks and categorize chats

### 🔗 MCP Integration

- **Model Context Client** Utilizes the same MCP config JSON format as Claude Desktop- Use ANY tools!
- **Model Context Protocol Server** for external tool integration
- **Standardized AI Agent Communication** following MCP specifications
- **Tool Ecosystem Compatibility** with other MCP-enabled systems
- **Bidirectional Data Exchange** with external AI agents

### 📊 Advanced Analytics

- **Emotional Trend Analysis** with stability metrics
- **Cognitive Pattern Recognition** and optimization
- **Personalized Recommendations** based on interaction history
- **Data Export** in multiple formats (JSON, CSV, etc.)

### Data Flow

1. **User Input** → Frontend → FastAPI
2. **Processing** → Vector DB Search → Context Retrieval
3. **AI Processing** → Gemini API → Response Generation
4. **State Updates** → Emotional/Cognitive Analysis → Pattern Storage
5. **Memory Storage** → Vector DB → Persistent Learning
6. **External Access** → MCP Server → Tool Integration

## 🧠 AI Thinking & Reasoning Transparency

### Thinking Extraction Capabilities
- **Real-time Reasoning Capture**: Extract and analyze AI thought processes during conversations
- **Thought Summarization**: Automatic generation of reasoning summaries for quick understanding
- **Cognitive Transparency**: Full visibility into how Aura approaches problems and makes decisions
- **Reasoning Metrics**: Detailed analytics on thinking patterns, processing time, and cognitive load

### Thinking Configuration
- **Thinking Budget**: Configurable reasoning depth (1024-32768 tokens)
- **Response Integration**: Optional inclusion of reasoning in user responses
- **Pattern Analysis**: Long-term analysis of reasoning patterns and cognitive development
- **Performance Optimization**: Thinking efficiency metrics and optimization recommendations

## 🎭 Emotional Intelligence System

### Supported Emotions
- **Basic**: Normal, Happy, Sad, Angry, Excited, Fear, Disgust, Surprise
- **Complex**: Joy, Love, Peace, Creativity, DeepMeditation
- **Combined**: Hope (Anticipation + Joy), Optimism, Awe, Remorse
- **Social**: RomanticLove, PlatonicLove, ParentalLove, Friendliness

### Neurological Correlations
- **Brainwave Patterns**: Alpha, Beta, Gamma, Theta, Delta
- **Neurotransmitters**: Dopamine, Serotonin, Oxytocin, GABA, Norepinephrine
- **NTK Layers**: Neural Tensor Kernel mapping for emotional states

## 🧠 ASEKE Cognitive Framework

### Components
- **KS** (Knowledge Substrate): Shared conversational context
- **CE** (Cognitive Energy): Mental effort and focus allocation
- **IS** (Information Structures): Ideas and concept patterns
- **KI** (Knowledge Integration): Learning and connection processes
- **KP** (Knowledge Propagation): Information sharing mechanisms
- **ESA** (Emotional State Algorithms): Emotional influence on processing
- **SDA** (Sociobiological Drives): Social dynamics and trust factors

## 📊 Analytics & Insights

### Emotional Analysis
- **Stability Metrics**: Emotional consistency over time
- **Dominant Patterns**: Most frequent emotional states
- **Transition Analysis**: Emotional state changes and triggers
- **Intensity Tracking**: Emotional intensity distribution
- **Brainwave Correlation**: Neural activity pattern analysis

### Cognitive Tracking
- **Focus Patterns**: ASEKE component utilization
- **Learning Efficiency**: Knowledge integration rates
- **Context Switching**: Cognitive flexibility metrics
- **Attention Allocation**: Cognitive energy distribution

## 🚦 Performance-

# Responses take some time to process depending on tasks, any coder wants to see if they can speed up the processes I would be grateful.

### Optimization

- Vector database indexing for fast searches
- Async processing for concurrent requests
- Cost and Memory-efficient local all-minilm vector embedding generation
- Autonomous sub-model background Focus gating and task processing for state updates and tool use
- Tool learning adapter
- [MemVid](https://github.com/Olow304/memvid) infinite QR code video long term memory!

### Monitoring
- Health check endpoint
- Performance metrics collection
- Error tracking and reporting
- Resource usage monitoring

### MCP Client now fully functional!!! Memvid integration attempted- still testing.

 I am not a coder so hopefully it sets up right if anyone tries it.

## 🚀 Install and Start back and front ends using 2 terminals

### Prerequisites
- Python 3.12+
- Google API Key (from [Google AI Studio](https://aistudio.google.com/app/apikey))
- At least 4GB RAM (for vector embeddings)
- 2GB+ storage space

### Installation

1. **Clone or Fork and Navigate**:
   ```bash
   cd /emotion_ai/aura_backend
   ```

 Uses uv- pyproject.toml and creates a .venv with python 3.12 --seed in the backend
2. **Run Setup Script**:
   ```bash
   ./setup.sh
   ```

3. **Configure Environment**:

# Copy the env example in the backend to .env
 I will try to streamline all of this into an OS agnostic app soon.

# It will pick up from your OS environment if the API key is configured. It should work if your OS key is set as GEMINI_API_KEY too

```bash
# Edit the .env file to use your existing key, sort of unneeded now I think.
echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" > .env
  ```

# Current backend .env settings:

```bash
# Aura Backend Configuration
# ==========================

# Google API Configuration
GOOGLE_API_KEY=your-google-api-key-here

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./aura_chroma_db
AURA_DATA_DIRECTORY=./aura_data

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO

# MCP Server Configuration
MCP_SERVER_NAME=aura-companion
MCP_SERVER_VERSION=1.0.0

# Security Configuration
CORS_ORIGINS=["http://localhost:5173", "http://localhost:3000"]

# Features Configuration
ENABLE_EMOTIONAL_ANALYSIS=true
ENABLE_COGNITIVE_TRACKING=true
ENABLE_VECTOR_SEARCH=true
ENABLE_FILE_EXPORTS=true

# AI Response Configuration
# gemini-2.5-flash-preview-05-20
AURA_MODEL=gemini-2.5-flash-preview-05-20
AURA_MAX_OUTPUT_TOKENS=1000000

# Autonomic System Configuration
# gemini-2.0-flash-lite
AURA_AUTONOMIC_MODEL=gemini-2.0-flash-lite
AURA_AUTONOMIC_MAX_OUTPUT_TOKENS=100000
AUTONOMIC_ENABLED=true
AUTONOMIC_TASK_THRESHOLD=medium  # low, medium, high

# Rate Limiting Configuration
AUTONOMIC_MAX_CONCURRENT_TASKS=12  # Optimal concurrency for 30 rpm limit
AUTONOMIC_RATE_LIMIT_RPM=25        # Requests per minute (safety margin below 30)
AUTONOMIC_RATE_LIMIT_RPD=1200      # Requests per day (safety margin below 1400)
AUTONOMIC_TIMEOUT_SECONDS=60       # Increased for higher concurrency

# Main Model Rate Limiting (user-configurable based on plan)
MAIN_MODEL_RATE_LIMIT_RPM=10       # Conservative default, increase based on user plan
MAIN_MODEL_RATE_LIMIT_RPD=500     # Daily limit for main model

# Queue Management
AUTONOMIC_QUEUE_MAX_SIZE=100       # Maximum queued tasks
AUTONOMIC_QUEUE_PRIORITY_ENABLED=true  # Enable priority-based processing

```

**Start Services**: This will start the backend, it runs in dev mode so hot restarts on saved code changes.
 I will try and design an internal copy that the Aura system can use to add to itself safely with a revert to the original system on failures feature. Sounds fun huh?

```bash
./start.sh
```

### Frontend activation

**Prerequisites:**  Node.js


1. Install dependencies:

```bash
npm install
```

2. Run the app, also in dev mode so it also hot restarts on code changes:

```bash
npm run dev
```

Go to-
Local:   http://localhost:5173

Network: use --host to expose

![alt text](image-5.png)


## 📡 API Endpoints

### Core API
- **Health Check**: `GET /health`
- **Process Conversation**: `POST /conversation`
- **Search Memories**: `POST /search`
- **Emotional Analysis**: `GET /emotional-analysis/{user_id}`
- **Export Data**: `POST /export/{user_id}`

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 🔗 MCP Integration

### Available MCP Tools- Working on emotional state records, hopefully fixed tomorrow

1. **search_aura_memories**: Semantic search through conversation history
2. **analyze_aura_emotional_patterns**: Deep emotional trend analysis
3. **store_aura_conversation**: Add memories to Aura's knowledge base
4. **get_aura_user_profile**: Retrieve user personalization data
5. **export_aura_user_data**: Data export functionality
6. **query_aura_emotional_states**: Information about emotional intelligence system
7. **query_aura_aseke_framework**: ASEKE cognitive architecture details

### Connecting External Tools

To connect external MCP clients to Aura:
# Example MCP client configuration- for Claude or other clients to talk to Aura or use as a system.
Edit your directory path and place in claude desktop config json.

```bash
{
  "mcpServers": {
    "aura-companion": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/ty/Repositories/ai_workspace/emotion_ai/aura_backend",
        "run",
        "aura_server.py"
      ]
    }
  }
}
```

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────┐
│                  Frontend                       │
│              (React/TypeScript)                 │
└─────────────────┬───────────────────────────────┘
                  │ HTTP/WebSocket
┌─────────────────▼───────────────────────────────┐
│                FastAPI                          │
│             (REST API Layer)                    │
├─────────────────┬───────────────────────────────┤
│                 │                               │
│  ┌──────────────▼─────────────┐                │
│  │     Vector Database        │                │
│  │       (ChromaDB)           │                │
│  │                            │                │
│  │ • Conversation Memory      │                │
│  │ • Emotional Patterns       │                │
│  │ • Cognitive States         │                │
│  │ • Knowledge Substrate      │                │
│  └────────────────────────────┘                │
│                                                 │
│  ┌────────────────────────────┐                │
│  │     State Manager          │                │
│  │                            │                │
│  │ • Emotional Transitions    │                │
│  │ • Cognitive Focus Changes  │                │
│  │ • Automated DB Operations  │                │
│  │ • Pattern Recognition      │                │
│  └────────────────────────────┘                │
│                                                 │
│  ┌────────────────────────────┐                │
│  │     File System            │                │
│  │                            │                │
│  │ • User Profiles            │                │
│  │ • Data Exports             │                │
│  │ • Session Storage          │                │
│  │ • Backup Management        │                │
│  └────────────────────────────┘                │
└─────────────────┬───────────────────────────────┘
                  │ MCP Protocol
┌─────────────────▼───────────────────────────────┐
│              MCP Server                         │
│         (External Tool Access)                 │
│                                                 │
│ • Memory Search Tools                           │
│ • Emotional Analysis Tools                      │
│ • Data Export Tools                             │
│ • ASEKE Framework Access                        │
└─────────────────────────────────────────────────┘
```


## 🧪 Testing

### Health Check (Working)
```bash
curl http://localhost:8000/health
```

### Thinking Functionality Tests (New!)
```bash
# Test thinking extraction capabilities
cd aura_backend
python test_thinking.py

# Interactive thinking demonstration
python thinking_demo.py

# Check thinking system status
curl http://localhost:8000/thinking-status
```

### Unit Tests
```bash
pytest tests/
```

### Integration Tests
```bash
./test_setup.py
```

### Load Testing
```bash
# Example using wrk
wrk -t12 -c400 -d30s http://localhost:8000/health
```

### Local Development
 I apologize for the mess, I do not know if any of this works below but feel free to try if you are brave or know what you are doing.

### Production (Docker)
```bash
# Build image
docker build -t aura-backend .

# Run container
docker run -p 8000:8000 -v ./aura_data:/app/aura_data aura-backend
```

### Systemd Service
```bash
# Copy service file
sudo cp aura-backend.service /etc/systemd/system/

# Enable and start
sudo systemctl enable aura-backend
sudo systemctl start aura-backend
```

## 🤝 Integration with Frontend

### API Endpoints to Update
Update your frontend to use these endpoints:

```typescript
const API_BASE = 'http://localhost:8000';

// Replace localStorage with API calls
const response = await fetch(`${API_BASE}/conversation`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    user_id: userId,
    message: userMessage,
    session_id: sessionId
  })
});
```

### WebSocket Support (Future)
Real-time updates and streaming responses will be available via WebSocket connections.

## 📚 Advanced Usage

### Custom MCP Tools
Create custom MCP tools by extending the `mcp_server.py`:

```python
@tool
async def custom_aura_tool(params: CustomParams) -> Dict[str, Any]:
    """Your custom tool implementation"""
    # Implementation here
    pass
```

### Vector Database Queries
Direct vector database access for advanced queries:

```python
from main import vector_db
results = await vector_db.search_conversations(
    query="emotional support",
    user_id="user123",
    n_results=10
)
```

![alt text](image-3.png)

## 🐛 Troubleshooting

### Common Issues

1. **Installation Errors**:
   ```bash
   # Ensure Python 3.12+
   python3 --version

   # Clean installation
   rm -rf venv/
   ./setup.sh
   ```

2. **API Key Issues**:
   ```bash
   # Check environment
   source venv/bin/activate
   echo $GOOGLE_API_KEY
   ```

3. **Vector DB Issues**: This is asshole AI- you will lose your db
   ```bash
   # Reset database
   rm -rf aura_chroma_db/
   ./test_setup.py
   ```

4. **Memory Issues**:
   - Increase system memory allocation
   - Reduce vector embedding batch sizes
   - Use lightweight embedding models

### Logs
Check logs in:
- Console output during development
- System logs: `journalctl -u aura-backend` (if using systemd)
- Application logs: `./aura_data/logs/`

## 🔒 Security- WARNING! AI Generated so I have 0 trust in these features

### Data Protection
- All user data stored locally
- No external data transmission (except Google API)
- Vector embeddings are anonymized
- Session data encrypted in transit

### Access Control
- API key authentication
- Rate limiting enabled
- CORS configuration
- Input validation and sanitization


## 🛣️ Roadmap

### Upcoming Features
- [ ] Real-time WebSocket connections
- [ ] Advanced emotion prediction models
- [ ] Multi-user collaboration features
- [ ] Enhanced MCP tool ecosystem
- [ ] Mobile app backend support
- [ ] Advanced analytics dashboard
- [ ] Integration with external AI models

### Long-term Vision
- Multi-modal interaction (voice, video, text)
- Federated learning across Aura instances
- Advanced personality adaptation
- Enterprise deployment options
- Open-source community ecosystem

## 📄 License

My stuff is MIT I suppose but there is other software like google-genai and memvid so it is a mixed bag I think
 ie don't steal my ideas and try to make money, without me. lol but I am super poor.

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests for review.

## 📞 Support

For issues and support:
1. Check troubleshooting section
2. Review logs and error messages
3. Create detailed issue reports
4. Join community discussions

---

**Aura Emotion AI** - *Powering the future of AI companionship and assistance through advanced emotional intelligence and sophisticated memory systems.*
