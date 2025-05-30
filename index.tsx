

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
  Learning: { fullName: "General Learning", description: "A general state of absorbing and processing information without a specific ASEKE focus."} // Fallback
};


// System instruction moved to backend


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
  intensity?: string; // Added for low/medium/high
  primaryComponents?: string[]; // Added for combined emotions
}

interface EmotionalStatesData {
  [emotionName: string]: EmotionalState;
}

const EMOTIONAL_STATES_DATA: EmotionalStatesData = {
  Normal: {
    Formula: "N(x) = R(x) AND C(x)", Components: { R: "Routine activities are being performed.", C: "No significant emotional triggers." },
    NTK_Layer: "Theta-like_NTK", Brainwave: "Alpha", Neurotransmitter: "Serotonin",
    Description: "A baseline state of calmness and routine engagement."
  },
  Excited: {
    Formula: "E(x) = A(x) OR S(x)", Components: { A: "Anticipation of a positive event.", S: "Stimulus exceeds a certain threshold." },
    NTK_Layer: "Gamma-like_NTK", Brainwave: "Beta", Neurotransmitter: "Dopamine",
    Description: "Feeling enthusiastic or eager, often in anticipation of something positive or due to high stimulation."
  },
  Angry: {
    Formula: "A(x) = I(x) AND NOT F(x)", Components: { I: "Injustice perceived.", F: "Fair resolution achieved." },
    NTK_Layer: "Beta-like_NTK", Brainwave: "Theta", Neurotransmitter: "Norepinephrine",
    Description: "Feeling strong displeasure or antagonism, often due to perceived injustice or threat."
  },
  Happy: {
    Formula: "H(x) = P(x) AND G(x)", Components: { P: "Positive events occurred.", G: "Goals achieved." },
    NTK_Layer: "Alpha-like_NTK", Brainwave: "Beta", Neurotransmitter: "Endorphin",
    Description: "Feeling pleased and content, often due to positive events or achieved goals."
  },
  Sad: {
    Formula: "S(x) = L(x) OR F(x)", Components: { L: "Loss experienced.", F: "Failure experienced." },
    NTK_Layer: "Theta-like_NTK", Brainwave: "Delta", Neurotransmitter: "Serotonin",
    Description: "Feeling sorrowful or unhappy, often due to loss or failure."
  },
  Joy: {
    Formula: "J(x) = H(x) AND E(x)", Components: { H: "Happiness sustained.", E: "Excitement present." },
    NTK_Layer: "Gamma-like_NTK", Brainwave: "Gamma", Neurotransmitter: "Oxytocin",
    Description: "Intense happiness and elation, often a combination of sustained happiness and excitement.",
    primaryComponents: ["Happy", "Excited"]
  },
  Peace: {
    Formula: "P(x) = NOT C(x) AND B(x)", Components: { C: "Conflict present.", B: "Balance in emotional state." },
    NTK_Layer: "Delta-like_NTK", Brainwave: "Theta", Neurotransmitter: "GABA",
    Description: "A state of tranquility and calm, free from conflict and with emotional balance."
  },
  RomanticLove: {
    Formula: "RL(x) = A(x) AND C(x)", Components: { A: "Attraction present.", C: "Commitment present." },
    NTK_Layer: "Alpha-like_NTK", Brainwave: "Delta", Neurotransmitter: "Oxytocin",
    Description: "Deep affection and care towards a romantic partner, involving attraction and commitment.",
    primaryComponents: ["Joy", "Trust"] // Example, aligns with Plutchik's Love
  },
  PlatonicLove: {
    Formula: "PL(x) = F(x) AND T(x)", Components: { F: "Friendship established.", T: "Trust present." },
    NTK_Layer: "Theta-like_NTK", Brainwave: "Alpha", Neurotransmitter: "Serotonin",
    Description: "Deep affection and care towards friends, based on established friendship and trust.",
    primaryComponents: ["Trust", "Friendliness"] // Example
  },
  ParentalLove: {
    Formula: "PaL(x) = C(x) AND P(x)", Components: { C: "Care provided.", P: "Protection offered." },
    NTK_Layer: "Delta-like_NTK", Brainwave: "Theta", Neurotransmitter: "Oxytocin",
    Description: "Deep affection, care, and protectiveness towards one's children.",
    primaryComponents: ["Joy", "Trust", "Sadness"] // Sadness if child is hurt
  },
  Creativity: {
    Formula: "Cr(x) = I(x) AND N(x)", Components: { I: "Innovation present.", N: "Novelty present." },
    NTK_Layer: "Gamma-like_NTK", Brainwave: "Gamma", Neurotransmitter: "Dopamine",
    Description: "Feeling inspired and inventive, engaging in novel and innovative thinking."
  },
  DeepMeditation: {
    Formula: "DM(x) = F(x) AND C(x)", Components: { F: "Focus sustained.", C: "Calmness achieved." },
    NTK_Layer: "Delta-like_NTK", Brainwave: "Delta", Neurotransmitter: "Serotonin",
    Description: "A state of profound calm and sustained focus, often achieved through meditative practices."
  },
  Friendliness: {
    Formula: "Fr(x) = K(x) AND O(x)", Components: { K: "Kindness shown.", O: "Openness to social interaction." },
    NTK_Layer: "Alpha-like_NTK", Brainwave: "Alpha", Neurotransmitter: "Endorphin",
    Description: "Being kind, warm, and open to social interaction."
  },
  Curiosity: {
    Formula: "Cu(x) = Q(x) AND E(x)", Components: { Q: "Questions generated.", E: "Eagerness to learn." },
    NTK_Layer: "Beta-like_NTK", Brainwave: "Beta", Neurotransmitter: "Dopamine",
    Description: "A strong desire to learn or know something, often accompanied by questions and eagerness."
  },
  Hope: { // New Combined Emotion
    Formula: "Anticipation + Joy", Components: { A: "Anticipation of a positive future.", J: "Present feeling of joy or potential for it." },
    NTK_Layer: "Beta-like_NTK", Brainwave: "Alpha", Neurotransmitter: "Dopamine",
    Description: "Feeling optimistic and expectant about a positive outcome.",
    primaryComponents: ["Anticipation", "Joy"]
  },
  Optimism: { // New Combined Emotion
    Formula: "Anticipation + Joy + Trust", Components: { A: "Looking forward to good things.", J: "Feeling content or happy.", T: "Belief in a positive future or ability to cope." },
    NTK_Layer: "Alpha-like_NTK", Brainwave: "Beta", Neurotransmitter: "Serotonin",
    Description: "A hopeful and confident outlook on the future or successful outcomes.",
    primaryComponents: ["Anticipation", "Joy", "Trust"]
  },
  Awe: { // New Combined Emotion
    Formula: "Fear + Surprise", Components: { F: "A sense of something vast or powerful.", S: "An element of the unexpected or overwhelming." },
    NTK_Layer: "Gamma-like_NTK", Brainwave: "Theta", Neurotransmitter: "Norepinephrine", // Can be mixed
    Description: "A feeling of reverential respect mixed with fear or wonder, often elicited by something sublime.",
    primaryComponents: ["Fear", "Surprise"]
  },
  Remorse: { // New Combined Emotion
    Formula: "Sadness + Disgust", Components: { S: "Regret or sorrow for past actions.", D: "Self-disappointment or disapproval of one's actions." },
    NTK_Layer: "Theta-like_NTK", Brainwave: "Delta", Neurotransmitter: "Serotonin",
    Description: "Deep regret and guilt over a past action.",
    primaryComponents: ["Sadness", "Disgust"]
  },
  Love: { // General Love, can refine if needed
    Formula: "Joy + Trust", Components: { J: "Warmth and happiness in connection.", T: "Deep sense of security and reliability." },
    NTK_Layer: "Alpha-like_NTK", Brainwave: "Alpha", Neurotransmitter: "Oxytocin",
    Description: "A strong feeling of deep affection, warmth, and care for someone.",
    primaryComponents: ["Joy", "Trust"]
  }
};

// --- Chat Initialization & Messaging Functions ---
async function initializeChat() {
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

function showTypingIndicator() {
  if (!typingIndicatorElement) {
    typingIndicatorElement = document.createElement('div');
    typingIndicatorElement.className = 'message-bubble aura typing-indicator';
    typingIndicatorElement.setAttribute('aria-label', 'Aura is typing');
    typingIndicatorElement.textContent = 'Aura is typing...';
    messageArea.appendChild(typingIndicatorElement);
    scrollToBottom();
  }
}

function removeTypingIndicator() {
  if (typingIndicatorElement) {
    typingIndicatorElement.remove();
    typingIndicatorElement = null;
  }
}

async function displayMessage(
  text: string,
  sender: 'user' | 'aura' | 'error',
  isStreaming: boolean = false
) {
  if (!isStreaming) {
    removeTypingIndicator();
  }

  const messageBubble = document.createElement('div');
  messageBubble.className = `message-bubble ${sender}`;
  messageBubble.setAttribute('role', 'log');
  messageBubble.innerHTML = await marked.parse(text);

  if (sender === 'aura' && isStreaming && currentAuraMessageElement) {
    currentAuraMessageElement.innerHTML = messageBubble.innerHTML;
  } else if (sender === 'aura' && !isStreaming) {
    if (currentAuraMessageElement) {
      currentAuraMessageElement.innerHTML = messageBubble.innerHTML;
      // Add message actions to the existing message element
      if (sender !== 'error') {
        addMessageActions(currentAuraMessageElement, text, sender);
      }
      currentAuraMessageElement = null;
    } else {
      messageArea.appendChild(messageBubble);
      // Add message actions to new message
      if (sender !== 'error') {
        addMessageActions(messageBubble, text, sender);
      }
    }
  }
  else {
    messageArea.appendChild(messageBubble);
    // Add message actions to user and aura messages (not error messages)
    if (sender !== 'error') {
      addMessageActions(messageBubble, text, sender);
    }
  }

  scrollToBottom();
}

function scrollToBottom() {
  messageArea.scrollTop = messageArea.scrollHeight;
}

function setFormDisabledState(disabled: boolean) {
  messageInput.disabled = disabled;
  sendButton.disabled = disabled;
}

async function extractUserNameFromNameInput(userInput: string): Promise<string | null> {
  // Simple client-side name extraction for fallback
  const input = userInput.toLowerCase().trim();
  
  // Look for common name patterns
  const namePatterns = [
    /(?:i'?m|my name is|i am|call me|i'm)\s+([a-zA-Z]+)/i,
    /^([a-zA-Z]+)(?:\s|$)/i  // Simple first word if it looks like a name
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

// Memory management is now handled by the backend
// Interaction summarization and storage occurs server-side


// --- Emotion Detection and Display ---
// Emotion detection is now handled by the backend

function updateAuraEmotionDisplay(emotionResult: { name: string; intensity?: string } | null) {
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

// Emotion and cognitive processing moved to backend

function updateAuraCognitiveFocusDisplay(focusCode: string | null) {
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

// Cognitive focus processing moved to backend


// --- Enhanced UI Features ---
function setupMemorySearchPanel() {
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

function setupEmotionalInsightsPanel() {
  // Connect to existing HTML elements instead of creating new ones
  const showInsightsButton = document.getElementById('show-insights');
  const insightsContent = document.getElementById('insights-content');
  
  if (showInsightsButton && insightsContent) {
    showInsightsButton.addEventListener('click', async () => {
      if (!userName || !backendConnected) return;
      
      try {
        showInsightsButton.textContent = 'Loading...';
        const analysis = await auraAPI.getEmotionalAnalysis(userName, 7);
        
        insightsContent.innerHTML = `
          <div class="insights-data">
            <p><strong>Period:</strong> Last ${analysis.period_days} days</p>
            <p><strong>Total Interactions:</strong> ${analysis.total_entries}</p>
            <p><strong>Emotional Stability:</strong> ${(analysis.emotional_stability * 100).toFixed(1)}%</p>
            ${analysis.dominant_emotions && analysis.dominant_emotions.length > 0 ? 
              `<p><strong>Dominant Emotions:</strong> ${analysis.dominant_emotions.map(([e, c]) => `${e} (${c}x)`).join(', ')}</p>` : 
              '<p><strong>Dominant Emotions:</strong> Not enough data</p>'
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
        showInsightsButton.textContent = 'View 7-Day Analysis';
      }
    });
  }
}

// --- Message Actions (Delete, Edit, Regenerate) ---
function addMessageActions(messageBubble: HTMLElement, messageText: string, sender: 'user' | 'aura') {
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
  
  // Add event listeners
  const deleteBtn = actionsDiv.querySelector('.delete-btn');
  const editBtn = actionsDiv.querySelector('.edit-btn');
  const regenerateBtn = actionsDiv.querySelector('.regenerate-btn');
  
  if (deleteBtn) {
    deleteBtn.addEventListener('click', () => {
      if (confirm('Delete this message?')) {
        messageBubble.remove();
      }
    });
  }
  
  if (editBtn) {
    editBtn.addEventListener('click', () => {
      editMessage(messageBubble, messageText);
    });
  }
  
  if (regenerateBtn) {
    regenerateBtn.addEventListener('click', () => {
      regenerateResponse(messageBubble);
    });
  }
}

function editMessage(messageBubble: HTMLElement, originalText: string) {
  const textArea = document.createElement('textarea');
  textArea.value = originalText;
  textArea.className = 'edit-textarea';
  textArea.style.width = '100%';
  textArea.style.minHeight = '60px';
  textArea.style.resize = 'vertical';
  
  const actionDiv = document.createElement('div');
  actionDiv.className = 'edit-actions';
  actionDiv.innerHTML = `
    <button class="action-btn save-btn">💾 Save</button>
    <button class="action-btn cancel-btn">❌ Cancel</button>
  `;
  
  const originalContent = messageBubble.innerHTML;
  messageBubble.innerHTML = '';
  messageBubble.appendChild(textArea);
  messageBubble.appendChild(actionDiv);
  
  textArea.focus();
  
  const saveBtn = actionDiv.querySelector('.save-btn');
  const cancelBtn = actionDiv.querySelector('.cancel-btn');
  
  saveBtn?.addEventListener('click', async () => {
    const newText = textArea.value.trim();
    if (newText && newText !== originalText) {
      // Update the message
      messageBubble.innerHTML = await marked.parse(newText);
      addMessageActions(messageBubble, newText, 'user');
      
      // Resend the conversation from this point
      // This would require more complex state management
      // For now, just update the display
    } else {
      messageBubble.innerHTML = originalContent;
    }
  });
  
  cancelBtn?.addEventListener('click', () => {
    messageBubble.innerHTML = originalContent;
  });
}

async function regenerateResponse(messageBubble: HTMLElement) {
  if (!userName || !backendConnected) return;
  
  // Find the user message that prompted this response
  const userMessage = findPreviousUserMessage(messageBubble);
  if (!userMessage) return;
  
  // Show regenerating indicator
  messageBubble.innerHTML = '<div class="typing-indicator">Regenerating response...</div>';
  
  try {
    // Send the same message again to get a new response
    const response = await auraAPI.sendMessage({
      user_id: userName,
      message: userMessage,
      session_id: currentSessionId || undefined
    });
    
    // Update the message content
    messageBubble.innerHTML = await marked.parse(response.response);
    addMessageActions(messageBubble, response.response, 'aura');
    
    // Update emotional and cognitive state
    updateAuraEmotionDisplay({
      name: response.emotional_state.name,
      intensity: response.emotional_state.intensity
    });
    updateAuraCognitiveFocusDisplay(response.cognitive_state.focus);
    
  } catch (error) {
    console.error('Failed to regenerate response:', error);
    messageBubble.innerHTML = '<div class="message-bubble error">Failed to regenerate response</div>';
  }
}

function findPreviousUserMessage(auraMessageBubble: HTMLElement): string | null {
  let currentElement = auraMessageBubble.previousElementSibling;
  
  while (currentElement) {
    if (currentElement.classList.contains('message-bubble') && 
        currentElement.classList.contains('user')) {
      return currentElement.textContent || null;
    }
    currentElement = currentElement.previousElementSibling;
  }
  
  return null;
}

// --- Main Initialization ---
async function main() {
  try {
    await initializeChat();
    
    // Setup enhanced UI features - connect to existing HTML elements
    setupMemorySearchPanel();
    setupEmotionalInsightsPanel();
    
  } catch (e) {
    console.error("Failed to initialize chat:", e);
    if (typeof displayMessage === "function") {
      await displayMessage(
        `Error initializing chat: ${e instanceof Error ? e.message : String(e)}`,
        'error'
      );
    } else {
      messageArea.innerHTML = `<div class="message-bubble error">Error initializing chat: ${e instanceof Error ? e.message : String(e)}</div>`;
    }
  }
}

main();
