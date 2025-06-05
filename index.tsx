/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { marked } from 'marked';
import { auraAPI, type ConversationResponse } from './src/services/auraApi';

// --- Session State ---
let currentSessionId: string | null = null;

// --- DOM Elements ---
const messageArea = document.getElementById('message-area') as HTMLElement;
const messageInput = document.getElementById('message-input') as HTMLInputElement;
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const sendButton = document.getElementById('send-button') as HTMLButtonElement;
const auraEmotionStatusElement = document.getElementById('aura-emotion-status') as HTMLElement;
const auraEmotionDetailsElement = document.getElementById('aura-emotion-details') as HTMLElement;
const auraCognitiveFocusElement = document.getElementById('aura-cognitive-focus') as HTMLElement;
const auraCognitiveFocusDetailsElement = document.getElementById('aura-cognitive-focus-details') as HTMLElement;
const themeToggle = document.getElementById('theme-toggle') as HTMLButtonElement;

// --- Chat State ---
let currentAuraMessageElement: HTMLElement | null = null;
let typingIndicatorElement: HTMLElement | null = null;
let userName: string | null = null;
let awaitingNameInput = false;
let backendConnected = false;

// --- ASEKE Cognitive Architecture Concepts ---
interface AsekeConcept {
  fullName: string;
  description: string;
}

const ASEKE_CONCEPTS: Record<string, AsekeConcept> = {
  KS: { fullName: "Knowledge Substrate", description: "The shared context, environment, and history of our discussion." },
  CE: { fullName: "Cognitive Energy", description: "The mental effort, attention, and focus being applied to the conversation." },
  IS: { fullName: "Information Structures", description: "The ideas, concepts, models, and patterns we are exploring or building." },
  KI: { fullName: "Knowledge Integration", description: "How new information is being connected with existing understanding and beliefs." },
  KP: { fullName: "Knowledge Propagation", description: "How ideas and information are being shared or potentially spread." },
  ESA: { fullName: "Emotional State Algorithms", description: "How feelings and emotions are influencing perception, valuation, and interaction." },
  SDA: { fullName: "Sociobiological Drives", description: "How social dynamics, trust, or group context might be shaping our interaction." },
  Learning: { fullName: "General Learning", description: "A general state of absorbing and processing information without a specific ASEKE focus." }
};

// --- Emotional States Data Structure ---
interface ComponentDetails {
  [key: string]: string;
}

interface EmotionalState {
  Formula: string;
  Components: ComponentDetails;
  NTK_Layer: string;
  Brainwave: string;
  Neurotransmitter: string;
  Description?: string;
  intensity?: string;
  primaryComponents?: string[];
}

interface EmotionalStatesData {
  [emotionName: string]: EmotionalState;
}

const EMOTIONAL_STATES_DATA: EmotionalStatesData = {
  Normal: {
    Formula: "N(x) = R(x) AND C(x)",
    Components: { R: "Routine activities are being performed.", C: "No significant emotional triggers." },
    NTK_Layer: "Theta-like_NTK",
    Brainwave: "Alpha",
    Neurotransmitter: "Serotonin",
    Description: "A baseline state of calmness and routine engagement."
  },
  Excited: {
    Formula: "E(x) = A(x) OR S(x)",
    Components: { A: "Anticipation of a positive event.", S: "Stimulus exceeds a certain threshold." },
    NTK_Layer: "Gamma-like_NTK",
    Brainwave: "Beta",
    Neurotransmitter: "Dopamine",
    Description: "Feeling enthusiastic or eager, often in anticipation of something positive or due to high stimulation."
  },
  Angry: {
    Formula: "A(x) = I(x) AND NOT F(x)",
    Components: { I: "Injustice perceived.", F: "Fair resolution achieved." },
    NTK_Layer: "Beta-like_NTK",
    Brainwave: "Theta",
    Neurotransmitter: "Norepinephrine",
    Description: "Feeling strong displeasure or antagonism, often due to perceived injustice or threat."
  },
  Happy: {
    Formula: "H(x) = P(x) AND G(x)",
    Components: { P: "Positive events occurred.", G: "Goals achieved." },
    NTK_Layer: "Alpha-like_NTK",
    Brainwave: "Beta",
    Neurotransmitter: "Endorphin",
    Description: "Feeling pleased and content, often due to positive events or achieved goals."
  },
  Sad: {
    Formula: "S(x) = L(x) OR F(x)",
    Components: { L: "Loss experienced.", F: "Failure experienced." },
    NTK_Layer: "Theta-like_NTK",
    Brainwave: "Delta",
    Neurotransmitter: "Serotonin",
    Description: "Feeling sorrowful or unhappy, often due to loss or failure."
  },
  Joy: {
    Formula: "J(x) = H(x) AND E(x)",
    Components: { H: "Happiness sustained.", E: "Excitement present." },
    NTK_Layer: "Gamma-like_NTK",
    Brainwave: "Gamma",
    Neurotransmitter: "Oxytocin",
    Description: "Intense happiness and elation, often a combination of sustained happiness and excitement.",
    primaryComponents: ["Happy", "Excited"]
  },
  Peace: {
    Formula: "P(x) = NOT C(x) AND B(x)",
    Components: { C: "Conflict present.", B: "Balance in emotional state." },
    NTK_Layer: "Delta-like_NTK",
    Brainwave: "Theta",
    Neurotransmitter: "GABA",
    Description: "A state of tranquility and calm, free from conflict and with emotional balance."
  },
  RomanticLove: {
    Formula: "RL(x) = A(x) AND C(x)",
    Components: { A: "Attraction present.", C: "Commitment present." },
    NTK_Layer: "Alpha-like_NTK",
    Brainwave: "Delta",
    Neurotransmitter: "Oxytocin",
    Description: "Deep affection and care towards a romantic partner, involving attraction and commitment.",
    primaryComponents: ["Joy", "Trust"]
  },
  PlatonicLove: {
    Formula: "PL(x) = F(x) AND T(x)",
    Components: { F: "Friendship established.", T: "Trust present." },
    NTK_Layer: "Theta-like_NTK",
    Brainwave: "Alpha",
    Neurotransmitter: "Serotonin",
    Description: "Deep affection and care towards friends, based on established friendship and trust.",
    primaryComponents: ["Trust", "Friendliness"]
  },
  ParentalLove: {
    Formula: "PaL(x) = C(x) AND P(x)",
    Components: { C: "Care provided.", P: "Protection offered." },
    NTK_Layer: "Delta-like_NTK",
    Brainwave: "Theta",
    Neurotransmitter: "Oxytocin",
    Description: "Deep affection, care, and protectiveness towards one's children.",
    primaryComponents: ["Joy", "Trust", "Sadness"]
  },
  Creativity: {
    Formula: "Cr(x) = I(x) AND N(x)",
    Components: { I: "Innovation present.", N: "Novelty present." },
    NTK_Layer: "Gamma-like_NTK",
    Brainwave: "Gamma",
    Neurotransmitter: "Dopamine",
    Description: "Feeling inspired and inventive, engaging in novel and innovative thinking."
  },
  DeepMeditation: {
    Formula: "DM(x) = F(x) AND C(x)",
    Components: { F: "Focus sustained.", C: "Calmness achieved." },
    NTK_Layer: "Delta-like_NTK",
    Brainwave: "Delta",
    Neurotransmitter: "Serotonin",
    Description: "A state of profound calm and sustained focus, often achieved through meditative practices."
  },
  Friendliness: {
    Formula: "Fr(x) = K(x) AND O(x)",
    Components: { K: "Kindness shown.", O: "Openness to social interaction." },
    NTK_Layer: "Alpha-like_NTK",
    Brainwave: "Alpha",
    Neurotransmitter: "Endorphin",
    Description: "Being kind, warm, and open to social interaction."
  },
  Curiosity: {
    Formula: "Cu(x) = Q(x) AND E(x)",
    Components: { Q: "Questions generated.", E: "Eagerness to learn." },
    NTK_Layer: "Beta-like_NTK",
    Brainwave: "Beta",
    Neurotransmitter: "Dopamine",
    Description: "A strong desire to learn or know something, often accompanied by questions and eagerness."
  },
  Hope: {
    Formula: "Anticipation + Joy",
    Components: { A: "Anticipation of a positive future.", J: "Present feeling of joy or potential for it." },
    NTK_Layer: "Beta-like_NTK",
    Brainwave: "Alpha",
    Neurotransmitter: "Dopamine",
    Description: "Feeling optimistic and expectant about a positive outcome.",
    primaryComponents: ["Anticipation", "Joy"]
  },
  Optimism: {
    Formula: "Anticipation + Joy + Trust",
    Components: { A: "Looking forward to good things.", J: "Feeling content or happy.", T: "Belief in a positive future or ability to cope." },
    NTK_Layer: "Alpha-like_NTK",
    Brainwave: "Beta",
    Neurotransmitter: "Serotonin",
    Description: "A hopeful and confident outlook on the future or successful outcomes.",
    primaryComponents: ["Anticipation", "Joy", "Trust"]
  },
  Awe: {
    Formula: "Fear + Surprise",
    Components: { F: "A sense of something vast or powerful.", S: "An element of the unexpected or overwhelming." },
    NTK_Layer: "Gamma-like_NTK",
    Brainwave: "Theta",
    Neurotransmitter: "Norepinephrine",
    Description: "A feeling of reverential respect mixed with fear or wonder, often elicited by something sublime.",
    primaryComponents: ["Fear", "Surprise"]
  },
  Remorse: {
    Formula: "Sadness + Disgust",
    Components: { S: "Regret or sorrow for past actions.", D: "Self-disappointment or disapproval of one's actions." },
    NTK_Layer: "Theta-like_NTK",
    Brainwave: "Delta",
    Neurotransmitter: "Serotonin",
    Description: "Deep regret and guilt over a past action.",
    primaryComponents: ["Sadness", "Disgust"]
  },
  Love: {
    Formula: "Joy + Trust",
    Components: { J: "Warmth and happiness in connection.", T: "Deep sense of security and reliability." },
    NTK_Layer: "Alpha-like_NTK",
    Brainwave: "Alpha",
    Neurotransmitter: "Oxytocin",
    Description: "A strong feeling of deep affection, warmth, and care for someone.",
    primaryComponents: ["Joy", "Trust"]
  }
};

// --- Chat Initialization & Messaging Functions ---
async function initializeChat(): Promise<void> {
  // Check backend health
  try {
    await auraAPI.healthCheck();
    backendConnected = true;
    console.log("✅ Backend connection established");
  } catch (error) {
    console.error("❌ Backend connection failed:", error);
    backendConnected = false;
    await displayMessage("⚠️ Unable to connect to Aura's advanced backend. Some features may be limited.", 'error');
  }

  // Get stored user name (we'll keep this in localStorage for simplicity)
  userName = localStorage.getItem('aura_user_name');
  let initialAuraGreeting = "";

  if (userName) {
    initialAuraGreeting = `Welcome back, ${userName}! I'm ready to continue our conversation with enhanced memory and emotional understanding. What would you like to explore today?`;
  } else {
    initialAuraGreeting = "Hello! I'm Aura, your Adaptive Reflective Companion. It seems we're establishing a new Knowledge Substrate (KS) together! What's your name, so I can personalize our interactions?";
    awaitingNameInput = true;
  }

  updateAuraEmotionDisplay({ name: "Normal", intensity: "Medium" });
  updateAuraCognitiveFocusDisplay("Learning");

  await displayMessage(initialAuraGreeting, 'aura', false);

  setFormDisabledState(false);
  messageInput.focus();
}

function showTypingIndicator(): void {
  if (!typingIndicatorElement) {
    typingIndicatorElement = document.createElement('div');
    typingIndicatorElement.className = 'message-bubble aura typing-indicator';
    typingIndicatorElement.setAttribute('aria-label', 'Aura is typing');
    typingIndicatorElement.textContent = 'Aura is typing...';
    messageArea.appendChild(typingIndicatorElement);
    scrollToBottom();
  }
}

function removeTypingIndicator(): void {
  if (typingIndicatorElement) {
    typingIndicatorElement.remove();
    typingIndicatorElement = null;
  }
}

async function displayMessage(
  text: string,
  sender: 'user' | 'aura' | 'error',
  isStreaming: boolean = false
): Promise<void> {
  if (!isStreaming) {
    removeTypingIndicator();
  }

  const messageBubble = document.createElement('div');
  messageBubble.className = `message-bubble ${sender}`;
  messageBubble.setAttribute('role', 'log');
  messageBubble.innerHTML = await marked.parse(text);

  // Simplified logic to prevent content loss
  if (sender === 'aura' && isStreaming && currentAuraMessageElement) {
    // Update streaming message
    currentAuraMessageElement.innerHTML = messageBubble.innerHTML;
  } else if (sender === 'aura' && !isStreaming && currentAuraMessageElement) {
    // Complete the streaming message
    currentAuraMessageElement.innerHTML = messageBubble.innerHTML;
    addMessageActions(currentAuraMessageElement, sender);
    currentAuraMessageElement = null;
  } else {
    // Add new message to area
    messageArea.appendChild(messageBubble);
    addMessageActions(messageBubble, sender);
  }

  scrollToBottom();
}

// Enhanced scroll function to ensure visibility of latest content
function scrollToBottom(): void {
  if (messageArea) {
    // Ensure DOM is fully updated before scrolling
    setTimeout(() => {
      // Force scroll to absolute bottom
      messageArea.scrollTop = messageArea.scrollHeight + 1000;
    }, 50);
  }
}

function setFormDisabledState(disabled: boolean): void {
  messageInput.disabled = disabled;
  sendButton.disabled = disabled;
}

async function extractUserNameFromNameInput(userInput: string): Promise<string | null> {
  // Simple client-side name extraction for fallback
  const input = userInput.toLowerCase().trim();

  // Look for common name patterns
  const namePatterns = [
    /(?:i'?m|my name is|i am|call me|i'm)s+([a-zA-Z]+)/i,
    /^([a-zA-Z]+)(?:s|$)/i  // Simple first word if it looks like a name
  ];

  for (const pattern of namePatterns) {
    const match = input.match(pattern);
    if (match && match[1] && match[1].length > 1 && match[1].length < 20) {
      const name = match[1].charAt(0).toUpperCase() + match[1].slice(1).toLowerCase();
      // Basic validation - avoid common non-names
      const nonNames = ['hello', 'hi', 'hey', 'yes', 'no', 'ok', 'okay', 'sure', 'good', 'fine'];
      if (!nonNames.includes(name.toLowerCase())) {
        return name;
      }
    }
  }

  return null;
}

// Chat form event listener
chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!userName && !awaitingNameInput) return;

  const userMessage = messageInput.value.trim();
  if (!userMessage) return;

  await displayMessage(userMessage, 'user');
  messageInput.value = '';
  setFormDisabledState(true);

  // Handle name input
  if (awaitingNameInput) {
    const extractedName = await extractUserNameFromNameInput(userMessage);

    if (extractedName) {
      userName = extractedName;
      localStorage.setItem('aura_user_name', userName);
      awaitingNameInput = false;

      const nameConfirmationMessage = `It's a pleasure to meet you, ${userName}! I'm now connecting to my advanced memory systems to provide you with a more personalized experience.`;
      await displayMessage(nameConfirmationMessage, 'aura', false);

      // Check if the message contains more than just the name
      const isJustName = userMessage.toLowerCase().includes(extractedName.toLowerCase()) && userMessage.length < extractedName.length + 15;

      if (!isJustName) {
        // Process the rest of the message
        // Continue to the conversation processing below
      } else {
        // Just the name, ready for next input
        setFormDisabledState(false);
        messageInput.focus();
        return;
      }
    } else {
      const repromptMessage = "I'm sorry, I didn't quite catch your name. Could you please tell me just your first name?";
      await displayMessage(repromptMessage, 'aura', false);
      setFormDisabledState(false);
      messageInput.focus();
      return;
    }
  }

  if (!userName) {
    console.error("No username available for conversation");
    await displayMessage("Please tell me your name first so I can personalize our conversation.", 'error');
    setFormDisabledState(false);
    return;
  }

  // Show typing indicator
  showTypingIndicator();

  try {
    if (!backendConnected) {
      // Fallback mode - basic response without advanced features
      removeTypingIndicator();
      await displayMessage("I'm currently running in basic mode due to backend connection issues. Advanced features like persistent memory and emotional analysis are temporarily unavailable.", 'aura', false);
      setFormDisabledState(false);
      messageInput.focus();
      return;
    }

    // Send message to backend API
    const response: ConversationResponse = await auraAPI.sendMessage({
      user_id: userName,
      message: userMessage,
      session_id: currentSessionId || undefined
    });

    // Update session ID
    currentSessionId = response.session_id;

    // Remove typing indicator and display response
    removeTypingIndicator();
    await displayMessage(response.response, 'aura', false);

    // Update emotional state display
    updateAuraEmotionDisplay({
      name: response.emotional_state.name,
      intensity: response.emotional_state.intensity
    });

    // Update cognitive focus display
    updateAuraCognitiveFocusDisplay(response.cognitive_state.focus);

    console.log(`🎭 Emotional state: ${response.emotional_state.name} (${response.emotional_state.intensity})`);
    console.log(`🧠 Cognitive focus: ${response.cognitive_state.focus}`);

    // Refresh chat history to show the new conversation
    await loadChatHistory();

  } catch (error) {
    console.error("Error communicating with Aura backend:", error);
    removeTypingIndicator();
    await displayMessage("I'm having trouble connecting to my advanced systems. Let me try a basic response.", 'error');

    // Could implement a basic fallback here if needed
  } finally {
    setFormDisabledState(false);
    messageInput.focus();
  }
});

// --- Emotion Detection and Display ---
function updateAuraEmotionDisplay(emotionResult: { name: string; intensity?: string } | null): void {
  if (!auraEmotionStatusElement || !auraEmotionDetailsElement) return;
  const currentEmotionName = emotionResult?.name || "Normal";
  const emotionIntensity = emotionResult?.intensity;

  const emotionData = EMOTIONAL_STATES_DATA[currentEmotionName];
  if (emotionData) {
    auraEmotionStatusElement.textContent = `${currentEmotionName}${emotionIntensity ? ` (${emotionIntensity})` : ''}`;
    let detailsText = `Brainwave: ${emotionData.Brainwave}`;
    if (emotionData.Neurotransmitter) {
      detailsText += ` | NT: ${emotionData.Neurotransmitter}`;
    }
    if (emotionData.primaryComponents && emotionData.primaryComponents.length > 0) {
        detailsText += ` | Components: ${emotionData.primaryComponents.join(', ')}`;
    }
    auraEmotionDetailsElement.textContent = detailsText;
  } else {
    auraEmotionStatusElement.textContent = "Normal";
    auraEmotionDetailsElement.textContent = `Brainwave: ${EMOTIONAL_STATES_DATA.Normal.Brainwave} | NT: ${EMOTIONAL_STATES_DATA.Normal.Neurotransmitter}`;
  }
}

function updateAuraCognitiveFocusDisplay(focusCode: string | null): void {
  if (!auraCognitiveFocusElement || !auraCognitiveFocusDetailsElement) return;
  const currentFocusCode = focusCode || "Learning";
  const focusData = ASEKE_CONCEPTS[currentFocusCode];

  if (focusData) {
    auraCognitiveFocusElement.textContent = `${focusData.fullName} (${currentFocusCode})`;
    auraCognitiveFocusDetailsElement.textContent = focusData.description;
  } else if (currentFocusCode === "Learning") {
    auraCognitiveFocusElement.textContent = "General Learning";
    auraCognitiveFocusDetailsElement.textContent = ASEKE_CONCEPTS.Learning.description;
  } else { // Fallback for unknown code, though detection should prevent this.
    auraCognitiveFocusElement.textContent = "Processing";
    auraCognitiveFocusDetailsElement.textContent = "Analyzing information.";
  }
}

// --- Enhanced UI Features ---
function setupMemorySearchPanel(): void {
  // Connect to existing HTML elements instead of creating new ones
  const searchButton = document.getElementById('search-memories');
  const memoryQuery = document.getElementById('memory-query') as HTMLInputElement;
  const memoryResults = document.getElementById('memory-results');

  if (searchButton && memoryQuery && memoryResults) {
    searchButton.addEventListener('click', async () => {
      if (!userName || !backendConnected) return;

      const query = memoryQuery.value.trim();
      if (!query) return;

      try {
        searchButton.textContent = 'Searching...';
        const response = await auraAPI.searchMemories(userName, query, 5);

        if (response.results && response.results.length > 0) {
          memoryResults.innerHTML = response.results.map(r => `
            <div class="memory-result">
              <div class="memory-content">${r.content}</div>
              <div class="memory-meta">Similarity: ${(r.similarity * 100).toFixed(1)}%</div>
            </div>
          `).join('');
        } else {
          memoryResults.innerHTML = '<div class="memory-result">No relevant memories found.</div>';
        }
      } catch (error) {
        console.error('Memory search failed:', error);
        memoryResults.innerHTML = '<div class="memory-result">Memory search temporarily unavailable.</div>';
      } finally {
        searchButton.textContent = 'Search';
      }
    });

    // Allow enter key to trigger search
    memoryQuery.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        searchButton.click();
      }
    });
  }
}

function setupChatHistoryPanel(): void {
    // Set up new chat button
    const newChatBtn = document.getElementById('new-chat-btn');

    if (newChatBtn) {
        newChatBtn.addEventListener('click', async () => {
            // Clear current chat
            const messageArea = document.getElementById('message-area');
            if (messageArea) {
                messageArea.innerHTML = '';
            }

            // Generate new session ID
            const newSessionId = crypto.randomUUID();
            currentSessionId = newSessionId;

            // Show welcome message with current date and time
            const now = new Date();
            const welcomeMessage = `Welcome back! It's ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}. How can I help you today?`;
            await displayMessage(welcomeMessage, 'aura');

            // Load chat history to update the list
            await loadChatHistory();
        });
    }

    // Load chat history on startup
    if (userName && backendConnected) {
        loadChatHistory();
    }
}

async function loadChatHistory(): Promise<void> {
    const chatHistoryList = document.getElementById('chat-history-list');
    if (!chatHistoryList || !userName || !backendConnected) return;

    try {
        const response = await auraAPI.getChatHistory(userName, 10);

        if (response.sessions && response.sessions.length > 0) {
            chatHistoryList.innerHTML = response.sessions.map((session: any) => {
                const date = new Date(session.last_time);
                const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                const firstMessage = session.messages[0]?.content || 'New conversation';
                const preview = firstMessage.length > 50 ? firstMessage.substring(0, 47) + '...' : firstMessage;

                return `
                    <div class="chat-session-item" data-session-id="${session.session_id}">
                        <div class="session-preview">${preview}</div>
                        <div class="session-time">${timeStr}</div>
                    </div>
                `;
            }).join('');

            // Add click handlers to load specific sessions
            document.querySelectorAll('.chat-session-item').forEach(item => {
                item.addEventListener('click', async (e) => {
                    const sessionId = (e.currentTarget as HTMLElement).dataset.sessionId;
                    if (sessionId) {
                        await loadChatSession(sessionId);
                    }
                });
            });
        } else {
            chatHistoryList.innerHTML = '<div class="no-history">No chat history yet</div>';
        }
    } catch (error) {
        console.error('Failed to load chat history:', error);
        chatHistoryList.innerHTML = '<div class="no-history">Failed to load history</div>';
    }
}

async function loadChatSession(sessionId: string): Promise<void> {
    if (!userName || !backendConnected) {
        await displayMessage("Unable to load chat session - not connected to backend", 'error');
        return;
    }

    try {
        // Show loading indicator
        const messageArea = document.getElementById('message-area');
        if (messageArea) {
            messageArea.innerHTML = '<div class="loading-session">Loading conversation history...</div>';
        }

        // Fetch chat history from backend
        const response = await auraAPI.getChatHistory(userName, 50); // Get more messages for session loading

        // Find the specific session
        const targetSession = response.sessions?.find((session: any) => session.session_id === sessionId);

        if (!targetSession) {
            await displayMessage(`Session ${sessionId} not found or has been removed.`, 'error');
            return;
        }

        // Clear message area
        if (messageArea) {
            messageArea.innerHTML = '';
        }

        // Set current session
        currentSessionId = sessionId;

        // Display session info
        const sessionDate = new Date(targetSession.start_time);
        const sessionInfo = `📅 **Chat Session**: ${sessionDate.toLocaleDateString()} at ${sessionDate.toLocaleTimeString()}`;
        await displayMessage(sessionInfo, 'aura');

        // Sort messages by timestamp to ensure chronological order
        const sortedMessages = targetSession.messages.sort((a: any, b: any) => 
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );

        // Display all messages from the session
        for (const message of sortedMessages) {
            const sender = message.sender === 'user' ? 'user' : 'aura';
            await displayMessage(message.content, sender);
            
            // Small delay to prevent UI overwhelming
            await new Promise(resolve => setTimeout(resolve, 100));
        }

        // Update emotional state if available from last Aura message
        const lastAuraMessage = sortedMessages.filter((msg: any) => msg.sender === 'aura').pop();
        if (lastAuraMessage && lastAuraMessage.emotion) {
            updateAuraEmotionDisplay({ 
                name: lastAuraMessage.emotion,
                intensity: 'Medium' // Default intensity since historical data may not have it
            });
        }

        console.log(`✅ Loaded session ${sessionId} with ${sortedMessages.length} messages`);

    } catch (error) {
        console.error('Failed to load chat session:', error);
        await displayMessage(`Failed to load session ${sessionId}. Please try again.`, 'error');
        
        // Restore to new chat if loading failed
        const newSessionId = crypto.randomUUID();
        currentSessionId = newSessionId;
        await displayMessage("Started a new conversation instead.", 'aura');
    }
}

function setupEmotionalInsightsPanel(): void {
  // Connect to existing HTML elements instead of creating new ones
  const showInsightsButton = document.getElementById('show-insights');
  const periodSelector = document.getElementById('insights-period') as HTMLSelectElement;
  const insightsContent = document.getElementById('insights-content');

  if (showInsightsButton && periodSelector && insightsContent) {
    showInsightsButton.addEventListener('click', async () => {
      if (!userName || !backendConnected) return;

      try {
        const selectedPeriod = periodSelector.value;
        showInsightsButton.textContent = 'Loading...';
        const analysis = await auraAPI.getEmotionalAnalysis(userName, selectedPeriod);

        // Format period display
        const periodText = selectedPeriod === 'hour' ? 'Hour' :
                          selectedPeriod === 'day' ? '24 Hours' :
                          selectedPeriod === 'week' ? 'Week' :
                          selectedPeriod === 'month' ? 'Month' :
                          selectedPeriod === 'year' ? 'Year' :
                          selectedPeriod === 'multi-year' ? '5 Years' :
                          `${analysis.period_days} days`;

        insightsContent.innerHTML = `
          <div class="insights-data">
            <p><strong>Period:</strong> Last ${periodText}</p>
            <p><strong>Total Interactions:</strong> ${analysis.total_entries}</p>
            <p><strong>Emotional Stability:</strong> ${(analysis.emotional_stability * 100).toFixed(1)}%</p>
            ${analysis.dominant_emotions && analysis.dominant_emotions.length > 0 ?
              `<p><strong>Dominant Emotions:</strong> ${analysis.dominant_emotions.map(([e, c]) => `${e} (${c}x)`).join(', ')}</p>` :
              '<p><strong>Dominant Emotions:</strong> Not enough data</p>'
            }
            ${analysis.brainwave_patterns && Object.keys(analysis.brainwave_patterns).length > 0 ?
              `<p><strong>Brainwave Patterns:</strong> ${Object.entries(analysis.brainwave_patterns).map(([wave, count]) => `${wave} (${count}x)`).join(', ')}</p>` :
              ''
            }
            <div class="recommendations">
              <h4>Recommendations:</h4>
              ${analysis.recommendations?.map(r => `<p>• ${r}</p>`).join('') || '<p>Continue building emotional awareness</p>'}
            </div>
          </div>
        `;
      } catch (error) {
        console.error('Failed to load emotional insights:', error);
        insightsContent.innerHTML = '<div class="insights-data">Emotional analysis temporarily unavailable.</div>';
      } finally {
        showInsightsButton.textContent = 'View Analysis';
      }
    });

    // Auto-refresh when period changes
    periodSelector.addEventListener('change', () => {
      if (insightsContent.innerHTML.includes('insights-data')) {
        showInsightsButton.click();
      }
    });
  }
}

// --- Message Actions (Delete, Edit, Regenerate) ---
function addMessageActions(messageBubble: HTMLElement, sender: 'user' | 'aura' | 'error'): void {
  // Don't add actions to error messages
  if (sender === 'error') return;

  const actionsDiv = document.createElement('div');
  actionsDiv.className = 'message-actions';
  actionsDiv.style.display = 'none';

  if (sender === 'user') {
    // Edit and Delete for user messages
    actionsDiv.innerHTML = `
      <button class="action-btn edit-btn" title="Edit message">✏️</button>
      <button class="action-btn delete-btn" title="Delete message">🗑️</button>
    `;
  } else {
    // Regenerate and Delete for Aura messages
    actionsDiv.innerHTML = `
      <button class="action-btn regenerate-btn" title="Regenerate response">🔄</button>
      <button class="action-btn delete-btn" title="Delete message">🗑️</button>
    `;
  }

  messageBubble.appendChild(actionsDiv);

  // Show actions on hover
  messageBubble.addEventListener('mouseenter', () => {
    actionsDiv.style.display = 'flex';
  });

  messageBubble.addEventListener('mouseleave', () => {
    actionsDiv.style.display = 'none';
  });

  // Add click handlers
  const deleteBtn = actionsDiv.querySelector('.delete-btn');
  const editBtn = actionsDiv.querySelector('.edit-btn');
  const regenerateBtn = actionsDiv.querySelector('.regenerate-btn');

  if (deleteBtn) {
    deleteBtn.addEventListener('click', () => {
      messageBubble.remove();
    });
  }

  if (editBtn) {
    editBtn.addEventListener('click', () => {
      // Edit functionality - create inline editor
      const originalContent = messageBubble.querySelector('.message-content') || messageBubble;
      const originalContentElement = originalContent as HTMLElement;
      originalContentElement.style.display = 'none';
      const originalText = originalContentElement.textContent || '';

      const editor = document.createElement('textarea');
      editor.className = 'message-editor';
      editor.value = originalText;
      editor.style.width = '100%';
      editor.style.minHeight = '60px';

      const saveBtn = document.createElement('button');
      saveBtn.textContent = 'Save';
      saveBtn.className = 'action-btn save-btn';

      const cancelBtn = document.createElement('button');
      cancelBtn.textContent = 'Cancel';
      cancelBtn.className = 'action-btn cancel-btn';

      const editorActions = document.createElement('div');
      editorActions.className = 'editor-actions';
      editorActions.appendChild(saveBtn);
      editorActions.appendChild(cancelBtn);

      originalContentElement.style.display = 'none';
      messageBubble.appendChild(editor);
      messageBubble.appendChild(editorActions);

      saveBtn.addEventListener('click', async () => {
        const newText = editor.value.trim();
        if (newText) {
          originalContentElement.innerHTML = await marked.parse(newText);
          // Could re-send to backend here if needed
        }
        editor.remove();
        editorActions.remove();
        originalContentElement.style.display = '';
      });

      cancelBtn.addEventListener('click', () => {
        editor.remove();
        editorActions.remove();
        originalContentElement.style.display = '';
      });

      editor.focus();
    });
  }

  if (regenerateBtn) {
    regenerateBtn.addEventListener('click', async () => {
      // Find the previous user message
      const allMessages = Array.from(messageArea.querySelectorAll('.message-bubble'));
      const currentIndex = allMessages.indexOf(messageBubble);

      if (currentIndex > 0) {
        const previousMessage = allMessages[currentIndex - 1];
        if (previousMessage.classList.contains('user')) {
          const userText = previousMessage.textContent || '';

          // Remove current Aura response
          messageBubble.remove();

          // Resend the message
          setFormDisabledState(true);
          showTypingIndicator();

          try {
            const response = await auraAPI.sendMessage({
              user_id: userName!,
              message: userText,
              session_id: currentSessionId || undefined
            });

            removeTypingIndicator();
            await displayMessage(response.response, 'aura', false);

            updateAuraEmotionDisplay({
              name: response.emotional_state.name,
              intensity: response.emotional_state.intensity
            });

            updateAuraCognitiveFocusDisplay(response.cognitive_state.focus);
          } catch (error) {
            console.error("Error regenerating response:", error);
            removeTypingIndicator();
            await displayMessage("I'm having trouble regenerating the response. Please try again.", 'error');
          } finally {
            setFormDisabledState(false);
            messageInput.focus();
          }
        }
      }
    });
  }
}

// --- Theme Toggle ---
themeToggle?.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('aura_theme', newTheme);
  themeToggle.textContent = newTheme === 'dark' ? '☀️' : '🌙';

  // Update toggle slider position
  const slider = themeToggle.querySelector('.theme-toggle-slider') as HTMLElement;
  if (slider) {
    if (newTheme === 'dark') {
      slider.style.transform = 'translateX(26px)';
    } else {
      slider.style.transform = 'translateX(0)';
    }
  }
});

// Load saved theme
const savedTheme = localStorage.getItem('aura_theme');
if (savedTheme) {
  document.documentElement.setAttribute('data-theme', savedTheme);
  if (themeToggle) {
    themeToggle.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    const slider = themeToggle.querySelector('.theme-toggle-slider') as HTMLElement;
    if (slider) {
      slider.style.transform = savedTheme === 'dark' ? 'translateX(26px)' : 'translateX(0)';
    }
  }
} else {
  // Default to dark theme as requested
  document.documentElement.setAttribute('data-theme', 'dark');
  if (themeToggle) {
    themeToggle.textContent = '☀️';
    const slider = themeToggle.querySelector('.theme-toggle-slider') as HTMLElement;
    if (slider) {
      slider.style.transform = 'translateX(26px)';
    }
  }
}

// --- Accessibility ---
messageInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

// --- Initialize Everything ---
document.addEventListener('DOMContentLoaded', () => {
  initializeChat();
  setupMemorySearchPanel();
  setupChatHistoryPanel();
  setupEmotionalInsightsPanel();
});

// --- Export for testing ---
export {
  initializeChat,
  displayMessage,
  updateAuraEmotionDisplay,
  updateAuraCognitiveFocusDisplay,
  EMOTIONAL_STATES_DATA,
  ASEKE_CONCEPTS
};
