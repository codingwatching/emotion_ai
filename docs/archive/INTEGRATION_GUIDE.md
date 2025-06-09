# Aura Frontend Integration Guide

This guide explains how to integrate the existing Aura frontend with the new advanced backend system.

## 🔄 Migration Overview

The new backend provides:
- **Vector database** for semantic memory instead of localStorage
- **Advanced emotional analysis** with pattern tracking
- **Cognitive state management** via ASEKE framework
- **MCP integration** for external tool access
- **Enhanced persistence** with file system operations

## 🚀 Quick Integration Steps

### 1. Backend Setup

First, ensure the backend is running:

```bash
cd /home/ty/Repositories/ai_workspace/emotion_ai/aura_backend
./setup.sh
# Edit .env with your Google API key
./start_all.sh
```

Verify backend is running:
- API: http://localhost:8000/health
- Docs: http://localhost:8000/docs

### 2. Frontend Modifications

#### A. Update API Configuration

Create a new API service file in your frontend:

```typescript
// src/services/auraApi.ts
const API_BASE_URL = 'http://localhost:8000';

export interface ConversationRequest {
  user_id: string;
  message: string;
  session_id?: string;
}

export interface ConversationResponse {
  response: string;
  emotional_state: {
    name: string;
    intensity: string;
    brainwave: string;
    neurotransmitter: string;
  };
  cognitive_state: {
    focus: string;
    description: string;
  };
  session_id: string;
}

export class AuraAPI {
  private static instance: AuraAPI;
  
  static getInstance(): AuraAPI {
    if (!AuraAPI.instance) {
      AuraAPI.instance = new AuraAPI();
    }
    return AuraAPI.instance;
  }

  async sendMessage(request: ConversationRequest): Promise<ConversationResponse> {
    const response = await fetch(`${API_BASE_URL}/conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  }

  async searchMemories(userId: string, query: string, nResults: number = 5) {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        query,
        n_results: nResults,
      }),
    });

    if (!response.ok) {
      throw new Error(`Search error: ${response.statusText}`);
    }

    return response.json();
  }

  async getEmotionalAnalysis(userId: string, days: number = 7) {
    const response = await fetch(`${API_BASE_URL}/emotional-analysis/${userId}?days=${days}`);
    
    if (!response.ok) {
      throw new Error(`Analysis error: ${response.statusText}`);
    }

    return response.json();
  }

  async exportUserData(userId: string, format: string = 'json') {
    const response = await fetch(`${API_BASE_URL}/export/${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ format }),
    });

    if (!response.ok) {
      throw new Error(`Export error: ${response.statusText}`);
    }

    return response.json();
  }

  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }
}
```

#### B. Replace localStorage with API Calls

Update your main chat logic (in `index.tsx`):

```typescript
// Replace the existing chat form submission logic
chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!userName) return;

  const userMessage = messageInput.value.trim();
  if (!userMessage) return;

  await displayMessage(userMessage, 'user');
  messageInput.value = '';
  setFormDisabledState(true);

  try {
    // Use the new API instead of direct Gemini calls
    const api = AuraAPI.getInstance();
    const response = await api.sendMessage({
      user_id: userName,
      message: userMessage,
      session_id: currentSessionId // Track session ID
    });

    // Update emotional state display
    updateAuraEmotionDisplay({
      name: response.emotional_state.name,
      intensity: response.emotional_state.intensity
    });

    // Update cognitive focus display
    updateAuraCognitiveFocusDisplay(response.cognitive_state.focus);

    // Display Aura's response
    await displayMessage(response.response, 'aura', false);

    // Store session ID for continuity
    currentSessionId = response.session_id;

  } catch (error) {
    console.error("Error communicating with Aura backend:", error);
    await displayMessage("Sorry, I'm having trouble connecting to my backend systems.", 'error');
  } finally {
    setFormDisabledState(false);
    messageInput.focus();
  }
});
```

#### C. Enhanced Memory Integration

Add memory search functionality:

```typescript
// Add a memory search feature
async function searchAuraMemories(query: string) {
  if (!userName) return;

  try {
    const api = AuraAPI.getInstance();
    const results = await api.searchMemories(userName, query, 5);
    
    // Display relevant memories in the chat
    if (results.results && results.results.length > 0) {
      const memoryContext = results.results
        .map(r => r.content)
        .join('\n\n');
      
      // Use this context to inform the conversation
      console.log('Relevant memories found:', memoryContext);
    }
  } catch (error) {
    console.error('Memory search failed:', error);
  }
}

// Add a search button to the UI
const searchButton = document.createElement('button');
searchButton.textContent = 'Search Memories';
searchButton.onclick = async () => {
  const query = prompt('What would you like me to remember?');
  if (query) {
    await searchAuraMemories(query);
  }
};
document.body.appendChild(searchButton);
```

#### D. Emotional Analysis Dashboard

Add an emotional insights panel:

```typescript
// Add emotional analysis feature
async function showEmotionalInsights() {
  if (!userName) return;

  try {
    const api = AuraAPI.getInstance();
    const analysis = await api.getEmotionalAnalysis(userName, 7);
    
    // Create a modal or panel to display insights
    const insightsModal = document.createElement('div');
    insightsModal.className = 'insights-modal';
    insightsModal.innerHTML = `
      <div class="insights-content">
        <h3>Your Emotional Patterns (Last 7 Days)</h3>
        <div class="insights-data">
          <p><strong>Emotional Stability:</strong> ${(analysis.emotional_stability * 100).toFixed(1)}%</p>
          <p><strong>Dominant Emotions:</strong> ${analysis.dominant_emotions?.map(([e, c]) => `${e} (${c}x)`).join(', ')}</p>
          <div class="recommendations">
            <h4>Recommendations:</h4>
            ${analysis.recommendations?.map(r => `<p>• ${r}</p>`).join('') || '<p>No specific recommendations</p>'}
          </div>
        </div>
        <button onclick="this.parentElement.parentElement.remove()">Close</button>
      </div>
    `;
    
    document.body.appendChild(insightsModal);
  } catch (error) {
    console.error('Failed to load emotional insights:', error);
  }
}

// Add insights button
const insightsButton = document.createElement('button');
insightsButton.textContent = 'Emotional Insights';
insightsButton.onclick = showEmotionalInsights;
document.body.appendChild(insightsButton);
```

### 3. Environment Configuration

Update your frontend `.env.local` file:

```bash
# Existing
GOOGLE_API_KEY=your_key_here

# New backend configuration
VITE_AURA_BACKEND_URL=http://localhost:8000
VITE_ENABLE_ADVANCED_FEATURES=true
VITE_ENABLE_MEMORY_SEARCH=true
VITE_ENABLE_EMOTIONAL_INSIGHTS=true
```

### 4. Enhanced UI Components

#### A. Memory Search Component

```typescript
// Add to your HTML
const memorySearchHTML = `
<div id="memory-search-panel" class="memory-panel">
  <h3>🔍 Memory Search</h3>
  <input type="text" id="memory-query" placeholder="Search conversations..." />
  <button id="search-memories">Search</button>
  <div id="memory-results"></div>
</div>
`;

// Add the component
document.getElementById('message-area')?.insertAdjacentHTML('beforebegin', memorySearchHTML);

// Add event listeners
document.getElementById('search-memories')?.addEventListener('click', async () => {
  const query = (document.getElementById('memory-query') as HTMLInputElement)?.value;
  if (query && userName) {
    const api = AuraAPI.getInstance();
    const results = await api.searchMemories(userName, query);
    
    const resultsDiv = document.getElementById('memory-results');
    if (resultsDiv) {
      resultsDiv.innerHTML = results.results.map(r => `
        <div class="memory-result">
          <div class="memory-content">${r.content}</div>
          <div class="memory-meta">Similarity: ${(r.similarity * 100).toFixed(1)}%</div>
        </div>
      `).join('');
    }
  }
});
```

#### B. Enhanced Emotional Display

```typescript
// Enhanced emotional state display with trends
function updateAuraEmotionDisplay(emotionResult: { name: string; intensity?: string }) {
  const currentEmotionName = emotionResult?.name || "Normal";
  const emotionIntensity = emotionResult?.intensity;

  // Update existing display
  if (auraEmotionStatusElement) {
    auraEmotionStatusElement.textContent = `${currentEmotionName}${emotionIntensity ? ` (${emotionIntensity})` : ''}`;
  }

  // Add trend information if available
  if (userName) {
    const api = AuraAPI.getInstance();
    api.getEmotionalAnalysis(userName, 1).then(analysis => {
      if (auraEmotionDetailsElement && analysis.dominant_emotions) {
        const trendInfo = `Recent: ${analysis.dominant_emotions[0]?.[0] || 'Normal'}`;
        auraEmotionDetailsElement.textContent += ` | ${trendInfo}`;
      }
    }).catch(console.error);
  }
}
```

### 5. CSS Styles for New Components

Add these styles to your `index.css`:

```css
/* Memory Search Panel */
.memory-panel {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 16px;
  backdrop-filter: blur(10px);
}

.memory-panel h3 {
  margin: 0 0 12px 0;
  color: #fff;
}

#memory-query {
  width: 70%;
  padding: 8px 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  margin-right: 8px;
}

#search-memories {
  padding: 8px 16px;
  background: linear-gradient(45deg, #667eea, #764ba2);
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: transform 0.2s;
}

#search-memories:hover {
  transform: translateY(-2px);
}

.memory-result {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border-left: 3px solid #667eea;
}

.memory-content {
  color: #fff;
  margin-bottom: 4px;
}

.memory-meta {
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.8em;
}

/* Insights Modal */
.insights-modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.insights-content {
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 16px;
  padding: 24px;
  max-width: 500px;
  max-height: 80vh;
  overflow-y: auto;
  color: white;
}

.insights-data {
  margin: 16px 0;
}

.recommendations {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px;
  margin-top: 12px;
}

/* Enhanced emotion display */
#aura-emotion-status {
  font-weight: bold;
  background: linear-gradient(45deg, #ff6b6b, #ee5757);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

#aura-cognitive-focus {
  font-weight: bold;
  background: linear-gradient(45deg, #4ecdc4, #44a08d);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
```

### 6. Testing the Integration

1. **Start the backend**: `./start_all.sh`
2. **Start the frontend**: `npm run dev`
3. **Test features**:
   - Send messages and verify they're stored in vector DB
   - Check emotional state updates
   - Try memory search functionality
   - View emotional insights

### 7. Advanced Features to Add

Once basic integration is working, consider adding:

#### A. WebSocket Connection (Future Enhancement)
```typescript
// WebSocket for real-time updates
const websocket = new WebSocket('ws://localhost:8000/ws');
websocket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};
```

#### B. Voice Integration Enhancement
```typescript
// Enhanced voice with backend integration
async function processVoiceInput(transcript: string) {
  const api = AuraAPI.getInstance();
  const response = await api.sendMessage({
    user_id: userName!,
    message: transcript,
    session_id: currentSessionId
  });
  
  // Use text-to-speech for response
  const utterance = new SpeechSynthesisUtterance(response.response);
  speechSynthesis.speak(utterance);
}
```

## 🔧 Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure the backend CORS settings include your frontend URL
2. **API Connection**: Verify the backend is running on port 8000
3. **Missing Dependencies**: Run `pip install -r requirements.txt` in the backend
4. **Environment Variables**: Check both frontend and backend .env files

### Debug Steps

1. Check backend health: `curl http://localhost:8000/health`
2. Monitor backend logs for errors
3. Use browser developer tools to check network requests
4. Verify API responses match expected format

## 🚀 Deployment Considerations

For production:

1. **Backend**: Use Docker or systemd service
2. **Frontend**: Update API URLs for production
3. **Security**: Add authentication, rate limiting
4. **Monitoring**: Set up logging and health checks

This integration transforms Aura from a simple chat interface into a sophisticated AI companion with persistent memory, emotional intelligence, and advanced analytics capabilities.
