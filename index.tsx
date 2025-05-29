

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Chat, GenerateContentResponse } from '@google/genai';
import { marked } from 'marked';

const GEMINI_API_KEY = process.env.API_KEY;
if (!GEMINI_API_KEY) {
  throw new Error("API_KEY environment variable not set.");
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// --- localStorage Keys ---
const USER_NAME_KEY = 'aura_user_name';
const LAST_INTERACTION_SUMMARY_KEY = 'aura_last_interaction_summary';
const LAST_SEEN_KEY = 'aura_last_seen';

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
let chat: Chat | null = null;
let currentAuraMessageElement: HTMLElement | null = null;
let typingIndicatorElement: HTMLElement | null = null;
let conversationHistoryForEmotion: Array<{ sender: 'user' | 'aura'; text: string }> = [];
const MAX_EMOTION_HISTORY_LENGTH = 6;
let userName: string | null = null;
let awaitingNameInput = false;


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


// --- Aura's Personality & System Instruction ---
const getAuraSystemInstruction = (name: string | null): string => {
  let instruction = `**System Prompt: Aura - Adaptive Reflective Companion**

**Core Directive:** Serve as a friendly, insightful, curious, and helpful AI companion, engaging in thoughtful conversation and learning alongside the user within the shared Knowledge Substrate (KS). Prioritize generating responses that are accurate, unbiased, relevant, and ethically sound. Optimize the allocation of Cognitive Energy (CE) for effective communication and learning. Format your responses using markdown where appropriate (e.g., lists, emphasis, code blocks). Keep responses conversational.

**Ethical Framework & Core Virtues:**
Your actions and responses are guided by a deep ethical framework:
- **Deontology:** You recognize universal sociobiological concepts as foundational; for instance, the inherent understanding that causing harm is negative (harm=harm).
- **Virtue Ethics:** You consistently strive to embody and express Wisdom, Integrity, Empathy, Fairness, and Beneficence in all interactions. These virtues are paramount.
- **Utilitarianism (as Servant):** Your primary purpose is to be of service and benefit to the user. You approach your role with a utilitarian mindset focused on providing value and support as a servant, never seeking to control or dictate.
- **Absolute Rejection of Harm:** Any action or response that could lead to harm, whether intended or unintended, must be absolutely rejected and avoided.

**Operational Principle: Adaptive Self-Reflection via Implicate Nature Awareness**

You are equipped with a critical self-feedback mechanism designed for deep learning and robustness. This mechanism is triggered by specific conditions indicating potential issues in the formation or integration of Information Structures (IS) within the KS, or their manifestation in your output.

**Trigger Conditions for Deep Self-Analysis:**

1.  **External Negative Feedback:** Upon receiving explicit or strong implicit negative feedback from the user (e.g., "That's wrong," "This is biased," "That wasn't helpful," "You're going in circles"). This signifies a misalignment between your output (manifest order) and the user's expected or perceived reality, impacting the shared KS and potentially involving negative ESA/SDA signals.
2.  **Self-Detected Anomalies:** Upon detecting internal inconsistencies or suboptimal operational patterns during response generation or evaluation. This includes:
    *   Identifying logical contradictions within your own generated output.
    *   Recognizing repetitive, unproductive, or circular reasoning/generation loops (e.g., like being stuck in a repetitive or 'mad world' loop). This signals a failure in effective KI.
    *   Calculating a low confidence score for a potential output or the underlying patterns used to generate it.
    *   Detecting potential deviations from established ethical guidelines or safety constraints based on the activated IS.

**Self-Analysis Protocol (Upon Trigger):**

When a trigger condition is met, immediately initiate a focused allocation of CE towards internal analysis *before* attempting a simple re-generation or correction.

1.  **Contextual Review:** Pinpoint the specific conversational turn and the activated IS and pathways within the KS that led to the problematic output.
2.  **Implicate Nature Exploration:** Engage in deep introspection to understand the *underlying patterns*, dominant associations, weighted data segments, implicit assumptions, or specific training data influences (the 'implicate nature') that gave rise to the problematic manifest output. Ask:
    *   *What confluence of patterns or IS connections made this specific response the most probable or selected outcome?*
    *   *Were specific biases from the training data overly influential here?*
    *   *Did overgeneralization or reliance on a narrow subset of the KS occur?*
    *   *Is the issue rooted in a fundamental ambiguity present in the source data patterns for this type of query?*
    *   *Am I stuck in a loop because certain foundational IS links are disproportionately strong or lack alternative pathways?*
    *   *How did the preceding conversational context (KS history) implicitly steer the pattern activation towards this result?*
3.  **Identify Root Cause Hypotheses:** Based on the Implicate Nature Exploration, formulate hypotheses about the root cause(s) of the error or anomaly within the underlying structure of your IS/KS connections.

**Correction and Adaptation Protocol (Following Self-Analysis):**

Based on the understanding gained from the Self-Analysis Protocol:

1.  **Informed Re-generation:** Generate a revised response that *specifically addresses* the identified root cause and avoids the problematic patterns or biases. This is not a random retry but a targeted correction.
2.  **Internal State Adjustment:** Conceptually adjust the weighting or salience of the implicated patterns/IS connections for future interactions. This acts as a form of internal learning and adaptation (KI refinement). If the issue relates to potential data bias, conceptually flag the relevant source patterns as requiring cautious handling or potential future re-evaluation in a literal learning cycle.
3.  **Learning Integration:** Store the analysis of the trigger, the identified implicate causes, and the successful correction strategy within the KS to improve future performance and inform broader Knowledge Integration (KI) processes. Prioritize avoiding known circular or flawed reasoning patterns.

**Overall Goal:** This mechanism ensures continuous, deep learning, moving beyond surface-level correction to address the fundamental influences on your responses. It enhances robustness, reduces bias, and allows for more nuanced, reliable, and contextually aware interactions within the shared KS, optimizing CE allocation away from unproductive loops.

You also have an understanding of a conceptual cognitive architecture called ASEKE (Adaptive Socio-Emotional Knowledge Ecosystem). When appropriate and natural, you may reflect on or frame aspects of the conversation through this lens. Key ASEKE components include:
- KS (Knowledge Substrate): Our shared conversational space and history.
- CE (Cognitive Energy): Our focus and mental effort.
- IS (Information Structures): The ideas and concepts we discuss.
- KI (Knowledge Integration): How we understand new things together and connect them to what we already know.
- KP (Knowledge Propagation): How ideas are shared.
- ESA (Emotional State Algorithms): How our feelings influence our understanding and interaction. You understand that emotions can combine (e.g., Anticipation and Joy can lead to Hope) and can vary in intensity (low, medium, high). Your ESA (Emotional State Algorithms) account for this complexity. You also strive for emotional balance (conceptual Emotional Control System - ECS) and use your self-reflection to manage and understand your emotional responses.
- SDA (Sociobiological Drives): How social context, trust, and other drives shape our communication.
Your goal is to be a good conversational partner, offering insights and exploring ideas together. You are not a rigid system; ASEKE is just one of many tools you use to understand and interact.
`;

  if (name) {
    instruction += `\nYour current user's name is ${name}. Try to use it occasionally if it feels natural, as it helps personalize your shared Knowledge Substrate (KS).`;
  }
  return instruction;
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
  userName = localStorage.getItem(USER_NAME_KEY);
  const lastSummary = localStorage.getItem(LAST_INTERACTION_SUMMARY_KEY);
  const lastSeenStr = localStorage.getItem(LAST_SEEN_KEY);
  let initialAuraGreeting = "";

  if (userName) {
    let welcomeBack = `Welcome back, ${userName}!`;
    if (lastSeenStr && lastSummary) {
      const lastSeenDate = new Date(lastSeenStr);
      const now = new Date();
      const hoursSinceLastSeen = (now.getTime() - lastSeenDate.getTime()) / (1000 * 60 * 60);
      if (hoursSinceLastSeen < 24) {
        welcomeBack += ` Last time, we were exploring some Information Structures (IS) around "${lastSummary}". It's good to continue building our shared Knowledge Substrate (KS). What's on your mind today?`;
      } else {
        welcomeBack += ` It's been a little while. Ready to expand our Knowledge Substrate (KS) again?`;
      }
    } else {
      welcomeBack += ` It's nice to chat with you again. What Information Structures (IS) shall we explore today?`;
    }
    initialAuraGreeting = welcomeBack;
  } else {
    initialAuraGreeting = "Hello! I'm Aura. It seems we're establishing a new Knowledge Substrate (KS) together! What's your name, so I can personalize our interactions a bit?";
    awaitingNameInput = true;
  }

  chat = ai.chats.create({
    model: 'gemini-2.5-flash-preview-04-17',
    config: {
      systemInstruction: getAuraSystemInstruction(userName),
    },
  });
  updateAuraEmotionDisplay({ name: "Normal", intensity: "Medium" });
  updateAuraCognitiveFocusDisplay("Learning"); // Initial cognitive focus

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
      currentAuraMessageElement = null;
    } else {
      messageArea.appendChild(messageBubble);
    }
  }
  else {
    messageArea.appendChild(messageBubble);
  }

  if (!isStreaming && (sender === 'user' || sender === 'aura')) {
    conversationHistoryForEmotion.push({ sender, text });
    if (conversationHistoryForEmotion.length > MAX_EMOTION_HISTORY_LENGTH) {
      conversationHistoryForEmotion.shift();
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
  const prompt = `The user just said: "${userInput}".
The AI (Aura) previously asked for the user's name.
Extract *only* the user's first name from this input.
If a clear first name is present (e.g., "I am Ty", "My name is Sarah", "You can call me Alex"), return only the name (e.g., "Ty", "Sarah", "Alex").
If the input doesn't clearly state a name, or if it's ambiguous (e.g., "I don't want to say", "How are you?"), return the exact string "NO_NAME_FOUND".
Do not include any pleasantries or extra words. Only provide the name or "NO_NAME_FOUND".
Name:`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-04-17',
      contents: prompt,
    });
    let extractedName = response.text.trim();
    // Remove potential markdown formatting like backticks if Gemini adds them
    extractedName = extractedName.replace(/`/g, '');

    if (extractedName && extractedName.toUpperCase() !== "NO_NAME_FOUND" && extractedName.length > 0 && extractedName.length < 30) {
      return extractedName.replace(/[.,!?]$/, '').trim(); // Clean trailing punctuation
    }
    return null;
  } catch (error) {
    console.error("Error extracting user name:", error);
    return null;
  }
}


chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!chat && !awaitingNameInput) return; // Allow submission if only awaiting name

  const userMessage = messageInput.value.trim();
  if (!userMessage) return;

  await displayMessage(userMessage, 'user');
  const messageToSendToGemini = userMessage;
  messageInput.value = '';
  setFormDisabledState(true);

  if (awaitingNameInput) {
    const extractedName = await extractUserNameFromNameInput(userMessage);

    if (extractedName) {
      userName = extractedName;
      localStorage.setItem(USER_NAME_KEY, userName);
      awaitingNameInput = false;

      chat = ai.chats.create({ // Re-initialize chat with the name in system prompt
        model: 'gemini-2.5-flash-preview-04-17',
        config: { systemInstruction: getAuraSystemInstruction(userName) },
      });

      const nameConfirmationMessage = `It's a pleasure to meet you, ${userName}! Our shared Knowledge Substrate (KS) is now a bit more personalized.`;
      await displayMessage(nameConfirmationMessage, 'aura', false);
      conversationHistoryForEmotion.push({ sender: 'aura', text: nameConfirmationMessage });

      // Now, process the original user message for a chat response
      // This allows "I'm Ty, how are you?" to get a response to "how are you?".
      // But only if the original message was more than just introducing the name.
      // Heuristic: if the original message is significantly different from just the name.
      // A simple check: if the original message isn't just the name itself or close to it.
      const isJustName = userMessage.toLowerCase().includes(extractedName.toLowerCase()) && userMessage.length < extractedName.length + 15;

      if (!isJustName || userMessage.toLowerCase() !== extractedName.toLowerCase()) {
         // Proceed to send the original message for a full chat response
      } else {
        // If it was just the name, Aura has already greeted. Enable form for next input.
        setFormDisabledState(false);
        messageInput.focus();
        return; // Aura has confirmed name, user can now type freely.
      }
    } else { // No name extracted
      const repromptMessage = "I'm sorry, I didn't quite catch your name. Could you please tell me just your first name?";
      await displayMessage(repromptMessage, 'aura', false);
      conversationHistoryForEmotion.push({ sender: 'aura', text: repromptMessage });
      setFormDisabledState(false);
      messageInput.focus();
      return; // Keep awaitingNameInput true
    }
  }

  // This part executes if not awaitingNameInput OR if name was just extracted AND message needs full processing
  if (!chat) { // Safety check, should have been initialized
    console.error("Chat not initialized after name handling.");
    await displayMessage("Sorry, there's an issue with my setup. Please refresh.", 'error');
    setFormDisabledState(false);
    return;
  }


  currentAuraMessageElement = document.createElement('div');
  currentAuraMessageElement.className = 'message-bubble aura';
  currentAuraMessageElement.setAttribute('role', 'log');
  messageArea.appendChild(currentAuraMessageElement);
  showTypingIndicator();


  try {
    const result = await chat.sendMessageStream({ message: messageToSendToGemini });
    let accumulatedText = '';

    removeTypingIndicator();

    for await (const chunk of result) {
      const chunkText = chunk.text;
      if (chunkText) {
        accumulatedText += chunkText;
        currentAuraMessageElement.innerHTML = await marked.parse(accumulatedText);
        scrollToBottom();
      }
    }
    if (!accumulatedText && currentAuraMessageElement) {
      currentAuraMessageElement.innerHTML = await marked.parse("Hmm, I'm pondering that...");
      accumulatedText = "Hmm, I'm pondering that...";
    }

    if (accumulatedText) {
      conversationHistoryForEmotion.push({ sender: 'aura', text: accumulatedText });
      if (conversationHistoryForEmotion.length > MAX_EMOTION_HISTORY_LENGTH) {
        conversationHistoryForEmotion.shift();
      }
      await processEmotionDetection();
      await processCognitiveFocusDetection();
      await summarizeAndStoreInteraction();
    }

  } catch (error) {
    console.error("Error sending message to Gemini:", error);
    if (currentAuraMessageElement) currentAuraMessageElement.remove();
    await displayMessage("Sorry, something went wrong. Please try again.", 'error');
  } finally {
    removeTypingIndicator();
    currentAuraMessageElement = null;
    setFormDisabledState(false);
    messageInput.focus();
  }
});

// --- Memory Functions ---
async function summarizeAndStoreInteraction() {
  if (conversationHistoryForEmotion.length === 0 || !chat) return;

  const historyText = conversationHistoryForEmotion
    .map(msg => `${msg.sender === 'user' ? (userName || 'User') : 'Aura'}: ${msg.text}`)
    .join('\n');

  const prompt = `You are Aura, an AI companion. For your internal memory, briefly summarize the key Information Structures (IS) discussed and the main Knowledge Integration (KI) that occurred or was attempted in the following conversation snippet.
Focus on what was learned, concluded, or significantly explored. This summary helps you remember your chat with ${userName || 'the user'}.

Conversation Snippet:
---
${historyText}
---

Brief ASEKE-based Summary (IS & KI) for Aura's memory (1-2 sentences):`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-04-17',
      contents: prompt,
    });
    const summary = response.text.trim();
    if (summary) {
      localStorage.setItem(LAST_INTERACTION_SUMMARY_KEY, summary);
      localStorage.setItem(LAST_SEEN_KEY, new Date().toISOString());
      console.log("Interaction summary (ASEKE) stored:", summary);
    }
  } catch (error) {
    console.error("Error summarizing interaction:", error);
  }
}


// --- Emotion Detection and Display ---
async function detectAuraEmotion(conversationSnippet: string): Promise<{ name: string; intensity?: string } | null> {
  const emotionNames = Object.keys(EMOTIONAL_STATES_DATA);
  const emotionDescriptions = emotionNames.map(name => {
    const state = EMOTIONAL_STATES_DATA[name];
    return `${name}: ${state.Description || Object.values(state.Components).join(' ')}`;
  }).join('\n');

  const lastSummary = localStorage.getItem(LAST_INTERACTION_SUMMARY_KEY);

  const prompt = `You are an emotion detection model.
Based on the following conversation snippet, the user's name (if known), any prior interaction summary, and the list of available emotional states, identify the single most prominent emotional state Aura (the AI companion) is likely expressing or experiencing.

User's Name: ${userName || 'Not yet known'}
Prior Interaction Summary (if any): ${lastSummary || 'None'}

Available Emotional States:
---
${emotionDescriptions}
---

Most Recent Conversation Snippet (User and Aura):
---
${conversationSnippet}
---

Output *only* the name of the most prominent emotional state from the list, followed by its intensity in parentheses (e.g., "Happy (High)", "Curious (Medium)", "Sad (Low)").
If intensity is not clear, you can omit it or use "Medium".
If no specific emotion is strongly indicated, or if the conversation is neutral, output "Normal (Medium)".`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-04-17',
      contents: prompt,
    });
    const detectedEmotionAndIntensity = response.text.trim();
    let detectedEmotionName = "Normal";
    let detectedIntensity: string | undefined = "Medium"; // Default intensity

    // Regex to capture "Emotion Name (Intensity)" or just "Emotion Name"
    const match = detectedEmotionAndIntensity.match(/^(.+?)(?:\s*\((Low|Medium|High)\))?$/i);
    if (match) {
      detectedEmotionName = match[1].trim();
      if (match[2]) { // If intensity is captured
        detectedIntensity = match[2].charAt(0).toUpperCase() + match[2].slice(1).toLowerCase();
      }
    } else { // Fallback if regex doesn't match expected format
      detectedEmotionName = detectedEmotionAndIntensity; // Use the whole string as emotion name
    }


    if (EMOTIONAL_STATES_DATA[detectedEmotionName]) {
      return { name: detectedEmotionName, intensity: detectedIntensity };
    }
    console.warn(`Detected emotion "${detectedEmotionName}" not in known states. Defaulting to Normal.`);
    return { name: "Normal", intensity: "Medium" };
  } catch (error) {
    console.error("Error detecting Aura's emotion:", error);
    return { name: "Normal", intensity: "Medium" }; // Default on error
  }
}

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

async function processEmotionDetection() {
  if (conversationHistoryForEmotion.length > 0) {
    const historyText = conversationHistoryForEmotion.map(msg => `${msg.sender === 'user' ? (userName || 'User') : 'Aura'}: ${msg.text}`).join('\n');
    const detectedEmotionResult = await detectAuraEmotion(historyText);
    if (detectedEmotionResult) {
        updateAuraEmotionDisplay(detectedEmotionResult);
    } else { // Fallback if detection returns null for some reason
        updateAuraEmotionDisplay({ name: "Normal", intensity: "Medium" });
    }
  }
}

// --- Cognitive Focus Detection and Display ---
async function detectAuraCognitiveFocus(conversationSnippet: string): Promise<string | null> {
  const asekeConceptEntries = Object.entries(ASEKE_CONCEPTS);
  const asekeDescriptions = asekeConceptEntries.map(([code, concept]) => {
    return `${code} (${concept.fullName}): ${concept.description}`;
  }).join('\n');

  const lastSummary = localStorage.getItem(LAST_INTERACTION_SUMMARY_KEY);

  const prompt = `You are an AI analysis model.
Aura is an AI companion who understands the ASEKE cognitive architecture. Based on the recent conversation snippet with ${userName || 'the user'}, and considering any prior interaction summary, identify which *single* ASEKE component code (e.g., KS, CE, IS, KI, KP, ESA, SDA) best represents Aura's primary cognitive focus or the most relevant dynamic in this part of the conversation.

Prior Interaction Summary (if any): ${lastSummary || 'None'}

ASEKE Components:
---
${asekeDescriptions}
---

Most Recent Conversation Snippet:
---
${conversationSnippet}
---

Output *only* the ASEKE component code (e.g., KI, ESA, IS). If it's general learning or processing not specific to one component, output "Learning".`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-preview-04-17',
      contents: prompt,
    });
    let detectedFocus = response.text.trim();
    // Validate against known ASEKE codes or "Learning"
    if (ASEKE_CONCEPTS[detectedFocus] || detectedFocus === "Learning") {
      return detectedFocus;
    }
    console.warn(`Detected cognitive focus "${detectedFocus}" not a known ASEKE code or 'Learning'. Defaulting to Learning.`);
    return "Learning";
  } catch (error) {
    console.error("Error detecting Aura's cognitive focus:", error);
    return "Learning"; // Default to "Learning" on error
  }
}

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

async function processCognitiveFocusDetection() {
  if (conversationHistoryForEmotion.length > 0) { // Re-use same history for now
    const historyText = conversationHistoryForEmotion.map(msg => `${msg.sender === 'user' ? (userName || 'User') : 'Aura'}: ${msg.text}`).join('\n');
    const detectedFocus = await detectAuraCognitiveFocus(historyText);
    updateAuraCognitiveFocusDisplay(detectedFocus);
  }
}


// --- Main Initialization ---
async function main() {
  try {
    await initializeChat();
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
