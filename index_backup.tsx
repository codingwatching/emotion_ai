/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { marked } from 'marked';
import { AuraAPI, type ConversationResponse } from './src/services/auraApi';

// Initialize API instance
const auraAPI = AuraAPI.getInstance();

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
let isLoadingHistory = false;

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
  // Add other emotions as needed
};

// --- Chat Initialization & Messaging Functions ---
async function initializeChat(): Promise<void> {
  console.log("🚀 Initializing Aura Chat System...");

  // Check backend health first
  try {
    const healthResponse = await fetch('http://localhost:8000/health');
    backendConnected = healthResponse.ok;
    console.log(`🔗 Backend connection: ${backendConnected ? 'Connected' : 'Failed'}`);
  } catch (error) {
    console.warn('⚠️ Backend health check failed:', error);
    backendConnected = false;
  }

  // Get stored user name
  userName = localStorage.getItem('aura_user_name');
  console.log(`👤 User identification: ${userName ? `Found user: ${userName}` : 'No user found'}`);

  // Get current time for greeting
  const now = new Date();
  const timeString = now.toLocaleDateString() + ' at ' + now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

  let initialAuraGreeting = "";

  if (userName) {
    initialAuraGreeting = `**Welcome back, ${userName}!**

*${timeString}*

I'm Aura, your adaptive reflective companion. I'm ready to continue our journey together. What would you like to explore today?`;
  } else {
    awaitingNameInput = true;
    initialAuraGreeting = `**Hello! I'm Aura 🌟**

*${timeString}*

I'm your adaptive reflective companion, designed to learn and grow with you through meaningful conversations.

Before we begin, what would you like me to call you?`;
  }

  updateAuraEmotionDisplay({ name: "Normal", intensity: "Medium" });
  updateAuraCognitiveFocusDisplay("Learning");

  await displayMessage(initialAuraGreeting, 'aura', false);

  setFormDisabledState(false);
  messageInput.focus();

  console.log("✅ Chat initialization complete");
}

function showTypingIndicator(): void {
  if (!typingIndicatorElement) {
    typingIndicatorElement = document.createElement('div');
    typingIndicatorElement.className = 'typing-indicator';
    typingIndicatorElement.textContent = 'Aura is thinking...';
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

  if (sender === 'aura' && isStreaming && currentAuraMessageElement) {
    currentAuraMessageElement.innerHTML = await marked.parse(text);
  } else if (sender === 'aura' && !isStreaming && currentAuraMessageElement) {
    currentAuraMessageElement.innerHTML = await marked.parse(text);
    currentAuraMessageElement = null;
  } else {
    messageArea.appendChild(messageBubble);
    if (sender === 'aura') {
      currentAuraMessageElement = messageBubble;
    }
  }

  scrollToBottom();
}

function scrollToBottom(): void {
  if (messageArea) {
    messageArea.scrollTop = messageArea.scrollHeight;
  }
}

function setFormDisabledState(disabled: boolean): void {
  messageInput.disabled = disabled;
  sendButton.disabled = disabled;
}

async function extractUserNameFromNameInput(userInput: string): Promise<string | null> {
  const input = userInput.toLowerCase().trim();

  const namePatterns = [
    /(?:i'?m|my name is|i am|call me|i'm)\s+([a-zA-Z]+)/i,
    /^([a-zA-Z]+)(?:\s|$)/i
  ];

  for (const pattern of namePatterns) {
    const match = input.match(pattern);
    if (match && match[1]) {
      return match[1].charAt(0).toUpperCase() + match[1].slice(1).toLowerCase();
    }
  }

  return null;
}

// Chat form event listener
chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();

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

      const welcomeMessage = `Nice to meet you, ${userName}! I'll remember your name for future conversations. How can I help you today?`;
      await displayMessage(welcomeMessage, 'aura');
      setFormDisabledState(false);
      return;
    } else {
      const clarificationMessage = "I'd love to know what to call you! Could you tell me your name more clearly? For example: 'My name is Alex' or just 'Alex'.";
      await displayMessage(clarificationMessage, 'aura');
      setFormDisabledState(false);
      return;
    }
  }

  if (!userName) {
    awaitingNameInput = true;
    const nameRequestMessage = "Before we continue, what would you like me to call you?";
    await displayMessage(nameRequestMessage, 'aura');
    setFormDisabledState(false);
    return;
  }

  showTypingIndicator();

  try {
    const response = await auraAPI.sendMessage({
      user_id: userName,
      message: userMessage,
      session_id: currentSessionId || undefined
    });

    if (!currentSessionId && response.session_id) {
      currentSessionId = response.session_id;
    }

    updateAuraEmotionDisplay({
      name: response.emotional_state.name,
      intensity: response.emotional_state.intensity
    });

    updateAuraCognitiveFocusDisplay(response.cognitive_state.focus);

    await displayMessage(response.response, 'aura');

  } catch (error) {
    console.error('Error in chat:', error);
    const errorMessage = backendConnected
      ? "I'm having trouble processing that right now. Could you try rephrasing your message?"
      : "I'm having trouble connecting to my backend. Please check if the server is running.";

    await displayMessage(errorMessage, 'error');
  } finally {
    setFormDisabledState(false);
  }
});

// --- Emotion Detection and Display ---
function updateAuraEmotionDisplay(emotionResult: { name: string; intensity?: string } | null): void {
  if (!auraEmotionStatusElement || !auraEmotionDetailsElement) return;

  const currentEmotionName = emotionResult?.name || "Normal";
  const emotionIntensity = emotionResult?.intensity;

  const emotionData = EMOTIONAL_STATES_DATA[currentEmotionName];
  if (emotionData) {
    auraEmotionStatusElement.textContent = currentEmotionName;
    auraEmotionDetailsElement.textContent = `Brainwave: ${emotionData.Brainwave} | NT: ${emotionData.Neurotransmitter}`;
  } else {
    auraEmotionStatusElement.textContent = currentEmotionName;
    auraEmotionDetailsElement.textContent = `Intensity: ${emotionIntensity || 'Unknown'}`;
  }
}

function updateAuraCognitiveFocusDisplay(focusCode: string | null): void {
  if (!auraCognitiveFocusElement || !auraCognitiveFocusDetailsElement) return;

  const currentFocusCode = focusCode || "Learning";
  const focusData = ASEKE_CONCEPTS[currentFocusCode];

  if (focusData) {
    auraCognitiveFocusElement.textContent = focusData.fullName;
    auraCognitiveFocusDetailsElement.textContent = focusData.description;
  } else if (currentFocusCode === "Learning") {
    auraCognitiveFocusElement.textContent = "Learning";
    auraCognitiveFocusDetailsElement.textContent = "Processing information and expanding understanding";
  } else {
    auraCognitiveFocusElement.textContent = currentFocusCode;
    auraCognitiveFocusDetailsElement.textContent = "Exploring new cognitive territories";
  }
}

// --- Enhanced UI Features ---
function setupMemorySearchPanel(): void {
  console.log("🔧 [MEMORY_SYSTEM] Initializing unified memory search system...");

  const searchButton = document.getElementById('search-memories') as HTMLButtonElement;
  const memoryQuery = document.getElementById('memory-query') as HTMLInputElement;
  const memoryResults = document.getElementById('memory-results') as HTMLElement;
  const activeMemoryCheckbox = document.getElementById('search-active-memory') as HTMLInputElement;
  const videoArchivesCheckbox = document.getElementById('search-video-archives') as HTMLInputElement;

  if (searchButton && memoryQuery && memoryResults) {
    searchButton.addEventListener('click', async () => {
      const query = memoryQuery.value.trim();

      if (!userName || !backendConnected) {
        memoryResults.innerHTML = '<div class="memory-result error">Please ensure backend connection and user authentication.</div>';
        return;
      }

      if (!query) {
        memoryResults.innerHTML = '<div class="memory-result">Please enter a search query to proceed.</div>';
        return;
      }

      try {
        searchButton.textContent = 'Searching...';
        searchButton.disabled = true;

        const response = await auraAPI.searchMemories(userName, query, 10);

        if (response.results && response.results.length > 0) {
          memoryResults.innerHTML = response.results.map((r, index) => {
            const displayContent = r.content.replace(/[*#]/g, '').trim();
            const similarity = (r.similarity * 100).toFixed(1);

            return `
              <div class="memory-result" data-source="active">
                <div class="memory-content">${displayContent}</div>
                <div class="memory-meta">
                  Relevance: ${similarity}% | Source: Active Memory
                </div>
              </div>
            `;
          }).join('');

          memoryResults.innerHTML += `
            <div class="search-summary">
              <strong>Search Results:</strong> ${response.results.length} found
            </div>
          `;
        } else {
          memoryResults.innerHTML = `
            <div class="memory-result">
              <div class="memory-content">No results found for "${query}"</div>
              <div class="memory-meta">
                Try using different keywords or check your search terms.
              </div>
            </div>
          `;
        }

      } catch (error) {
        console.error('Memory search error:', error);

        memoryResults.innerHTML = `
          <div class="memory-result error">
            <div class="memory-content">Memory search failed: ${(error as Error).message || 'Unknown error'}</div>
            <div class="memory-meta">
              Backend Status: ${backendConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        `;
      } finally {
        searchButton.textContent = 'Search';
        searchButton.disabled = false;
      }
    });

    memoryQuery.addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !searchButton.disabled) {
        e.preventDefault();
        searchButton.click();
      }
    });

    memoryQuery.addEventListener('input', (e) => {
      if ((e.target as HTMLInputElement).value.trim() === '') {
        memoryResults.innerHTML = '';
      }
    });
  } else {
    console.warn('⚠️ Memory search elements not found in DOM');
  }
}

function setupChatHistoryPanel(): void {
  console.log("🔧 [CHAT_HISTORY] Initializing chat history management system...");

  const newChatBtn = document.getElementById('new-chat-btn');

  if (newChatBtn) {
    newChatBtn.addEventListener('click', async () => {
      console.log("🔄 [NEW_CHAT] Initiating new chat session creation...");

      try {
        await performCompleteSystemReset();

        const newSessionId = crypto.randomUUID();
        currentSessionId = newSessionId;
        console.log(`📝 [NEW_CHAT] Session ID generated: ${newSessionId}`);

        await initializeNewChatSession();

        console.log("✅ [NEW_CHAT] New chat session successfully created");
      } catch (error) {
        console.error("❌ [NEW_CHAT] Failed to create new chat session:", error);
        await displayMessage("Error creating new chat. Please refresh the page.", 'error');
      }
    });
  }
}

async function performCompleteSystemReset(): Promise<void> {
  console.log("🧹 [SYSTEM_RESET] Performing complete chat state reset...");

  const messageArea = document.getElementById('message-area');
  if (messageArea) {
    messageArea.innerHTML = '';
  }

  currentAuraMessageElement = null;
  typingIndicatorElement = null;

  removeTypingIndicator();

  document.querySelectorAll('.chat-session-item').forEach(item => {
    item.classList.remove('active');
  });

  setFormDisabledState(false);

  console.log("✅ [SYSTEM_RESET] Chat state reset completed");
}

async function initializeNewChatSession(): Promise<void> {
  console.log("🚀 [SESSION_INIT] Initializing new chat session interface...");

  const now = new Date();
  const timeString = now.toLocaleDateString() + ' at ' + now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

  const welcomeMessage = `**Welcome back, ${userName || 'there'}!**

*${timeString}*

I'm ready to help you with your projects. What would you like to work on today?`;

  await displayMessage(welcomeMessage, 'aura');

  if (messageInput) {
    messageInput.focus();
  }

  console.log("✅ [SESSION_INIT] New session interface initialized");
}

function setupEmotionalInsightsPanel(): void {
  console.log("🔧 [INSIGHTS] Initializing emotional insights panel...");

  const showInsightsBtn = document.getElementById('show-insights') as HTMLButtonElement;
  const insightsPeriod = document.getElementById('insights-period') as HTMLSelectElement;
  const insightsContent = document.getElementById('insights-content') as HTMLElement;

  if (showInsightsBtn && insightsPeriod && insightsContent) {
    showInsightsBtn.addEventListener('click', async () => {
      if (!userName || !backendConnected) {
        insightsContent.innerHTML = '<div class="insights-data">Please ensure backend connection and user authentication.</div>';
        return;
      }

      try {
        showInsightsBtn.textContent = 'Analyzing...';
        showInsightsBtn.disabled = true;

        // For now, show a placeholder message
        insightsContent.innerHTML = `
          <div class="insights-data">
            <p><strong>Emotional Analysis:</strong> Coming soon!</p>
            <p>This feature will provide insights into your emotional patterns over the selected time period.</p>
          </div>
        `;

      } catch (error) {
        console.error('Insights error:', error);
        insightsContent.innerHTML = `
          <div class="insights-data error">
            Failed to load insights: ${(error as Error).message || 'Unknown error'}
          </div>
        `;
      } finally {
        showInsightsBtn.textContent = 'View Analysis';
        showInsightsBtn.disabled = false;
      }
    });
  }
}

// --- Theme Toggle ---
themeToggle?.addEventListener('click', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('aura_theme', newTheme);
});

// Load saved theme
const savedTheme = localStorage.getItem('aura_theme');
if (savedTheme) {
  document.documentElement.setAttribute('data-theme', savedTheme);
} else {
  document.documentElement.setAttribute('data-theme', 'dark');
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
  console.log('🎯 DOM Content Loaded - Starting Aura initialization');

  initializeChat();
  setupMemorySearchPanel();
  setupChatHistoryPanel();
  setupEmotionalInsightsPanel();

  console.log('✅ All systems initialized');
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
