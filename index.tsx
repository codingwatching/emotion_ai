/**
 * Aura Frontend UI Manager
 */

import { marked } from 'marked';
import { AuraAPI, ConversationResponse, EmotionalState, CognitiveState } from './src/services/auraApi';

// ============================================================================
// CONFIGURATION & TYPES
// ============================================================================

interface ChatSession {
  session_id: string;
  last_message: string;
  message_count: number;
  timestamp: string;
}

// ============================================================================
// GLOBAL STATE MANAGEMENT
// ============================================================================

class AuraUIManager {
  private api: AuraAPI;
  private currentSessionId: string | null = null;
  private userName: string | null = null;
  private awaitingNameInput = false;
  private backendConnected = false;
  private typingIndicatorElement: HTMLElement | null = null;
  private chatSessions: ChatSession[] = [];

  // DOM Elements
  private messageArea!: HTMLElement;
  private messageInput!: HTMLInputElement;
  private chatForm!: HTMLFormElement;
  private sendButton!: HTMLButtonElement;
  private emotionStatusElement!: HTMLElement;
  private emotionDetailsElement!: HTMLElement;
  private cognitiveFocusElement!: HTMLElement;
  private cognitiveFocusDetailsElement!: HTMLElement;
  private chatHistoryList!: HTMLElement;
  private leftPanel!: HTMLElement;
  private rightPanel!: HTMLElement;

  constructor(apiInstance?: AuraAPI) {
    this.api = apiInstance ?? AuraAPI.getInstance();
  }

  public async initialize(): Promise<void> {
    console.log("🚀 Initializing Aura UI Manager...");
    await this.initializeDOM();
    await this.checkBackendHealth();
    await this.loadUserData();
    this.setupUI();
    this.initializeEventListeners();
    await this.startChat();
    console.log("✅ Aura UI Manager initialized successfully.");
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================

  private async initializeDOM(): Promise<void> {
    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve));
    }

    // Get essential DOM elements
    this.messageArea = this.getRequiredElement('message-area');
    this.messageInput = this.getRequiredElement('message-input') as HTMLInputElement;
    this.chatForm = this.getRequiredElement('chat-form') as HTMLFormElement;
    this.sendButton = this.getRequiredElement('send-button') as HTMLButtonElement;
    this.emotionStatusElement = this.getRequiredElement('aura-emotion-status');
    this.emotionDetailsElement = this.getRequiredElement('aura-emotion-details');
    this.cognitiveFocusElement = this.getRequiredElement('aura-cognitive-focus');
    this.cognitiveFocusDetailsElement = this.getRequiredElement('aura-cognitive-focus-details');
    this.chatHistoryList = this.getRequiredElement('chat-history-list');
    this.leftPanel = this.getRequiredElement('left-panel');
    this.rightPanel = this.getRequiredElement('right-panel');

    console.log("✅ DOM elements initialized");
  }

  private getRequiredElement(id: string): HTMLElement {
    const element = document.getElementById(id);
    if (!element) {
      throw new Error(`Required element not found: ${id}`);
    }
    return element;
  }

  private async checkBackendHealth(): Promise<void> {
    try {
      console.log("🔍 Checking backend health...");
      const healthData = await this.api.healthCheck();
      this.backendConnected = true;
      console.log("✅ Backend connected:", healthData);

      // Test if the backend is actually responding to conversation endpoint
      try {
        console.log("🧪 Testing backend endpoints...");
        // We'll test this during the first actual message
      } catch (testError) {
        console.warn("⚠️ Backend health check passed but endpoints may not be working:", testError);
      }

    } catch (error) {
      console.warn("⚠️ Backend health check failed:", error);
      this.backendConnected = false;
      this.showConnectionWarning();
    }
  }

  private async loadUserData(): Promise<void> {
    const storedName = localStorage.getItem('auraUserName');
    if (storedName) {
      this.userName = storedName;
      this.awaitingNameInput = false; // User is known
      console.log(`User data loaded from localStorage: ${this.userName}`);
    } else {
      this.userName = null; // Explicitly null if not found
      this.awaitingNameInput = true; // Aura needs to ask for the name
      console.log("No user data found in local storage. Aura will ask for name.");
    }
  }

  // ============================================================================
  // UI SETUP
  // ============================================================================

  private setupUI(): void {
    this.setupTheme();
    this.setupMobileMenu();
    this.setupMemorySearch();
    this.setupEmotionalInsights();
    this.setupChatHistory();
    this.setupKeyboardShortcuts();
    this.updateVideoArchiveStatus();
  }

  private setupTheme(): void {
    const themeToggle = document.getElementById('theme-toggle');
    const savedTheme = localStorage.getItem('aura_theme') || 'dark';

    document.documentElement.setAttribute('data-theme', savedTheme);

    themeToggle?.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('aura_theme', newTheme);
    });
  }

  private setupMobileMenu(): void {
    const leftMenuBtn = document.getElementById('left-menu-btn');
    const rightMenuBtn = document.getElementById('right-menu-btn');

    leftMenuBtn?.addEventListener('click', () => {
      this.leftPanel.classList.toggle('collapsed');
    });

    rightMenuBtn?.addEventListener('click', () => {
      this.rightPanel.classList.toggle('collapsed');
    });

    // Close panels when clicking outside on mobile
    document.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      if (window.innerWidth <= 1024) {
        if (!this.leftPanel.contains(target) && !leftMenuBtn?.contains(target)) {
          this.leftPanel.classList.add('collapsed');
        }
        if (!this.rightPanel.contains(target) && !rightMenuBtn?.contains(target)) {
          this.rightPanel.classList.add('collapsed');
        }
      }
    });
  }

  private setupChatHistory(): void {
    const newChatBtn = document.getElementById('new-chat-btn');
    newChatBtn?.addEventListener('click', () => this.createNewChat());

    // Load chat history initially
    this.loadChatHistory();
    // Chat history will now be refreshed only when a chat is created or selected.
  }

  private setupMemorySearch(): void {
    const searchInput = document.getElementById('unified-search-input') as HTMLInputElement | null;
    const searchButton = document.getElementById('unified-search-button') as HTMLButtonElement | null;
    const searchResultsArea = document.getElementById('unified-search-results');
    const searchErrorElement = document.getElementById('unified-search-error'); // Element from your screenshot

    if (!searchInput || !searchButton || !searchResultsArea || !searchErrorElement) {
      console.error('Memory search UI elements not found. Search functionality will be disabled.');
      if (searchErrorElement) searchErrorElement.textContent = 'Search UI failed to load.';
      return;
    }

    // Initially disable search button if user is not known
    searchButton.disabled = !this.userName;

    searchButton.addEventListener('click', () => {
      const query = searchInput.value.trim();
      searchErrorElement.textContent = ''; // Clear previous errors
      searchResultsArea.innerHTML = '';   // Clear previous results

      if (!this.userName) {
        searchErrorElement.textContent = 'Error: User not identified. Please tell Aura your name in the chat.';
        console.warn('Memory search attempted without user identification.');
        return;
      }

      if (!query) {
        searchErrorElement.textContent = 'Please enter a search query.';
        return;
      }

      searchResultsArea.innerHTML = '<p>Searching memories...</p>';

      (async () => {
        try {
          console.log(`Searching memories for user "${this.userName}" with query "${query}"`);
          const response = await this.api.searchMemories(this.userName!, query);

          this.displaySearchResults(response, searchResultsArea);
        } catch (error: any) {
          console.error('Error during memory search:', error);
          searchErrorElement.textContent = `Search failed: ${error.message || 'Unknown error'}`;
          searchResultsArea.innerHTML = '<p>An error occurred during the search.</p>';
        }
      })();
    });
    console.log("✅ Unified Memory Search UI setup complete.");
  }

  private setupEmotionalInsights(): void {
    const showInsightsBtn = document.getElementById('show-insights');
    const insightsPeriod = document.getElementById('insights-period') as HTMLSelectElement;
    const insightsContent = document.getElementById('insights-content');

    if (!showInsightsBtn || !insightsPeriod || !insightsContent) return;

    showInsightsBtn.addEventListener('click', async () => {
      console.log(`📊 Emotional insights requested:`, { userName: this.userName, backendConnected: this.backendConnected });

      if (!this.userName || !this.backendConnected) {
        const errorMsg = !this.userName ? 'User not identified' : 'Backend not connected';
        insightsContent.innerHTML = `<div class="insights-data error">Error: ${errorMsg}</div>`;
        console.warn('⚠️ Emotional insights failed:', { userName: this.userName, backendConnected: this.backendConnected });
        return;
      }

      try {
        showInsightsBtn.textContent = 'Analyzing...';
        (showInsightsBtn as HTMLButtonElement).disabled = true;
        insightsContent.innerHTML = '<div class="insights-data">Analyzing emotional patterns...</div>';

        const period = insightsPeriod.value;
        console.log(`📤 Requesting emotional analysis for user: ${this.userName}, period: ${period}`);

        const analysis = await this.api.getEmotionalAnalysis(this.userName, period);
        console.log(`📥 Emotional analysis response:`, analysis);

        this.displayEmotionalInsights(analysis, insightsContent);

      } catch (error) {
        console.error('❌ Insights error:', error);
        insightsContent.innerHTML = `<div class="insights-data error">Failed to load insights: ${(error as Error).message}</div>`;
      } finally {
        showInsightsBtn.textContent = 'View Analysis';
        (showInsightsBtn as HTMLButtonElement).disabled = false;
      }
    });
  }

  private setupKeyboardShortcuts(): void {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Enter to send message
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        this.chatForm.dispatchEvent(new Event('submit'));
      }

      // Escape to close mobile panels
      if (e.key === 'Escape') {
        this.leftPanel.classList.add('collapsed');
        this.rightPanel.classList.add('collapsed');
      }
    });

    // Fixed message input handling - properly trigger form submission
    this.messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        // Trigger form submission event instead of calling handler directly
        this.chatForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
      }
    });
  }

  // ============================================================================
  // CHAT FUNCTIONALITY
  // ============================================================================

  private async startChat(): Promise<void> {
    try {
      const now = new Date();
      const timeString = `${now.toLocaleDateString()} at ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;

      let initialGreeting: string;

      if (this.userName) {
        initialGreeting = `**Welcome back, ${this.userName}!**\n\n*${timeString}*\n\nI'm ready to continue our conversation. What would you like to explore today?`;
      } else {
        this.awaitingNameInput = true;
        initialGreeting = `**Hello! I'm Aura 🌟**\n\n*${timeString}*\n\nI'm your adaptive reflective companion. Before we begin, what would you like me to call you?`;
      }

      this.updateEmotionalState({ name: "Normal", intensity: "Medium", brainwave: "Alpha", neurotransmitter: "Serotonin" });
      this.updateCognitiveState({ focus: "Learning", description: "Ready to assist and learn together" });

      await this.displayMessage(initialGreeting, 'aura');
      this.setFormState(false);
      this.messageInput.focus();

    } catch (error) {
      console.error("❌ Failed to start chat:", error);
      await this.displayMessage("I'm having trouble starting up. Please refresh the page if the issue persists.", 'error');
    }
  }

  private initializeEventListeners(): void {
    if (!this.chatForm) {
      console.error("❌ Chat form not found during event listener initialization.");
      // Optionally, throw an error or show a critical UI error
      this.showCriticalError("Chat input form failed to initialize. Please refresh.");
      return;
    }

    // Directly attach the submit listener to the class's chatForm property
    this.chatForm.addEventListener('submit', (event) => this.handleFormSubmit(event));
    console.log("✅ Chat form submit listener attached.");

    // You can add other global or essential listeners here if needed,
    // ensuring their respective DOM elements are also checked like this.chatForm.
  }

  private async handleFormSubmit(event: Event): Promise<void> {
    event.preventDefault();

    const userMessage = this.messageInput.value.trim();
    if (!userMessage) return;

    await this.displayMessage(userMessage, 'user');
    this.messageInput.value = '';
    this.setFormState(true);

    // Check for special commands
    if (await this.handleSpecialCommands(userMessage)) {
      return;
    }

    // Handle name input flow
    if (this.awaitingNameInput) {
      await this.handleNameInput(userMessage);
      return;
    }

    if (!this.userName) {
      this.awaitingNameInput = true;
      await this.displayMessage("Before we continue, what would you like me to call you?", 'aura');
      this.setFormState(false);
      return;
    }

    await this.processMessage(userMessage);
  }

  // ============================================================================
  // SPECIAL COMMAND HANDLING
  // ============================================================================

  private async handleSpecialCommands(message: string): Promise<boolean> {
    const lowerMessage = message.toLowerCase().trim();

    // Command to change name
    if (lowerMessage.startsWith('/name ') || lowerMessage.startsWith('/setname ')) {
      const newName = message.substring(message.indexOf(' ') + 1).trim();
      if (newName && newName.length >= 2) {
        const extractedName = this.extractNameFromInput(newName) || newName.split(/\s+/)[0];
        const oldName = this.userName;
        this.userName = extractedName;
        localStorage.setItem('auraUserName', this.userName);

        await this.displayMessage(`Great! I've updated your name to ${this.userName}.`, 'aura');

        // Notify the backend about the name change so it can use existing user data
        if (oldName !== this.userName) {
          try {
            console.log(`🔄 Notifying backend of name change: "${oldName}" → "${this.userName}"`);

            // Send a special message to backend to update user context
            const nameChangeRequest = {
              user_id: this.userName,
              message: `[SYSTEM] User name updated from "${oldName || 'unknown'}" to "${this.userName}". Please use existing data for "${this.userName}" if available.`,
              session_id: this.currentSessionId || undefined
            };

            const response = await this.api.sendMessage(nameChangeRequest);
            console.log(`✅ Backend notified of name change, response: ${response.response}`);

            // Update session ID if needed
            if (response.session_id && !this.currentSessionId) {
              this.currentSessionId = response.session_id;
            }

          } catch (error) {
            console.warn(`⚠️ Failed to notify backend of name change: ${error}`);
            await this.displayMessage("Name updated locally, but there may be an issue syncing with your chat history.", 'aura');
          }
        }

        this.updateUserSpecificUI();
        this.setFormState(false);
        return true;
      } else {
        await this.displayMessage("Please provide a valid name. Example: `/name Ty`", 'aura');
        this.setFormState(false);
        return true;
      }
    }

    // Command to reset name
    if (lowerMessage === '/resetname' || lowerMessage === '/changename') {
      this.userName = null;
      localStorage.removeItem('auraUserName');
      this.awaitingNameInput = true;
      await this.displayMessage("I've cleared your name. What would you like me to call you?", 'aura');
      this.updateUserSpecificUI();
      this.setFormState(false);
      return true;
    }

    return false; // Not a special command
  }

  private async handleNameInput(userInput: string): Promise<void> {
    const trimmedInput = userInput.trim();
    if (!trimmedInput) {
      await this.displayMessage("I need a name to call you by. Could you please tell me?", 'aura');
      this.awaitingNameInput = true;
      return;
    }

    // Extract name from conversational input
    let extractedName = this.extractNameFromInput(trimmedInput);

    if (!extractedName) {
      await this.displayMessage("I didn't catch your name clearly. Could you tell me just your name or nickname?", 'aura');
      this.awaitingNameInput = true;
      return;
    }

    this.userName = extractedName;
    this.awaitingNameInput = false;
    localStorage.setItem('auraUserName', this.userName);
    await this.displayMessage(`Great, I'll call you ${this.userName}! How can I help you today?`, 'aura');
    console.log(`User name set to: ${this.userName} and saved to localStorage.`);
    this.updateUserSpecificUI(); // Update UI elements that depend on the user name
  }

  private extractNameFromInput(input: string): string | null {
    const trimmed = input.trim();

    // If it's just a simple name (1-2 words, no complex grammar), use it directly
    const words = trimmed.split(/\s+/);
    if (words.length <= 2 && !/^(hi|hello|hey|my|name|is|i'm|im|call|me)\b/i.test(trimmed)) {
      return trimmed;
    }

    // Common patterns for name introduction
    const patterns = [
      /(?:my name is|i'm|im|i am|call me|name's)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)/i,
      /(?:hi|hello|hey)(?:\s+aura)?[,\s]*(?:my name is|i'm|im|i am)?\s*([a-zA-Z]+)/i,
      /^([a-zA-Z]+)(?:\s+here|$)/i // "Ty here" or just "Ty"
    ];

    for (const pattern of patterns) {
      const match = trimmed.match(pattern);
      if (match && match[1]) {
        // Take only the first word of the captured name to avoid titles/extra words
        const extractedName = match[1].trim().split(/\s+/)[0];
        // Make sure it's a reasonable name (not a common word)
        if (extractedName.length >= 2 && !/^(aura|hi|hello|hey|name|call|me|my|is|am)$/i.test(extractedName)) {
          return extractedName;
        }
      }
    }

    return null; // Couldn't extract a clear name
  }

  private updateUserSpecificUI(): void {
    // Example: Enable/disable search button based on userName
    const searchButton = document.getElementById('unified-search-button') as HTMLButtonElement | null;
    if (searchButton) {
      searchButton.disabled = !this.userName;
    }
    // The following assumes the existence of an element with id 'user-name-display-area' in the DOM.
    // If this element is missing, the update will silently fail.
    // To ensure robustness, provide a fallback warning in the console.
    const userNameDisplayElement = document.getElementById('user-name-display-area');
    if (userNameDisplayElement) {
      userNameDisplayElement.textContent = this.userName ? `User: ${this.userName}` : 'User: Not identified';
    } else {
      console.warn("user-name-display-area element not found in the DOM. Username display will not be updated.");
    }

    if (!this.userName) {
      const searchInput = document.getElementById('unified-search-input') as HTMLInputElement | null;
      if (searchInput) searchInput.value = '';

      const searchResultsArea = document.getElementById('unified-search-results');
      if (searchResultsArea) searchResultsArea.innerHTML = '';

      const searchErrorElement = document.getElementById('unified-search-error');
      if (searchErrorElement) searchErrorElement.textContent = 'User not identified. Please tell Aura your name in the chat.';
    }
  }

  private async processMessage(userMessage: string): Promise<void> {
    this.showTypingIndicator();

    try {
      console.log(`🤖 Sending message to backend:`, {
        userName: this.userName,
        userMessage,
        sessionId: this.currentSessionId,
        backendConnected: this.backendConnected
      });

      if (!this.backendConnected) {
        throw new Error('Backend connection not available');
      }

      if (!this.userName) {
        throw new Error('User name not set');
      }

      const requestData = {
        user_id: this.userName,
        message: userMessage,
        session_id: this.currentSessionId || undefined
      };

      console.log(`📤 Request data:`, requestData);

      const response: ConversationResponse = await this.api.sendMessage(requestData);

      console.log(`✅ Received response:`, response);

      // Validate response
      if (!response) {
        throw new Error('No response received from backend');
      }

      if (!response.response) {
        throw new Error('Invalid response: missing response text');
      }

      // Update session ID if new
      if (!this.currentSessionId && response.session_id) {
        this.currentSessionId = response.session_id;
        console.log(`📝 Session ID set to: ${this.currentSessionId}`);

        // Refresh chat history to show new session
        setTimeout(() => this.loadChatHistory(), 1000);
      }

      // Update UI states with validation
      if (response.emotional_state) {
        this.updateEmotionalState(response.emotional_state);
      } else {
        console.warn('⚠️ No emotional state in response');
      }

      if (response.cognitive_state) {
        this.updateCognitiveState(response.cognitive_state);
      } else {
        console.warn('⚠️ No cognitive state in response');
      }

      await this.displayMessage(response.response, 'aura');

    } catch (error) {
      console.error('❌ Chat error details:', {
        error,
        message: (error as Error).message,
        stack: (error as Error).stack,
        userName: this.userName,
        backendConnected: this.backendConnected
      });

      let errorMessage: string;
      const errorStr = (error as Error).message;

      // Handle specific database errors
      if (errorStr.includes('disk I/O error') || errorStr.includes('database')) {
        errorMessage = "🚨 Database connection issue detected. The conversation system is experiencing technical difficulties. Please try restarting the application.";
      } else if (errorStr.includes('ChromaDB') || errorStr.includes('instance')) {
        errorMessage = "⚙️ Database configuration conflict detected. Multiple instances may be running. Please restart the application.";
      } else if (!this.backendConnected) {
        errorMessage = "🔌 Backend connection lost. Please check if the server is running and refresh the page.";
      } else {
        errorMessage = `💥 Processing error: ${errorStr}`;
      }

      await this.displayMessage(errorMessage, 'error');

      // If it's a database error, also show a recovery suggestion
      if (errorStr.includes('disk I/O error') || errorStr.includes('ChromaDB')) {
        setTimeout(async () => {
          await this.displayMessage(
            "💡 **Recovery Suggestion**: This appears to be a database issue. Try:\n\n" +
            "1. Restart the entire application (backend + frontend)\n" +
            "2. Check if multiple ChromaDB instances are running\n" +
            "3. Ensure sufficient disk space\n" +
            "4. Check file permissions on the database directory",
            'aura'
          );
        }, 1000);
      }
    } finally {
      this.setFormState(false);
    }
  }

  // ============================================================================
  // UI UPDATES
  // ============================================================================

  private showTypingIndicator(): void {
    if (this.typingIndicatorElement) return;

    this.typingIndicatorElement = document.createElement('div');
    this.typingIndicatorElement.className = 'typing-indicator';
    this.typingIndicatorElement.innerHTML = `
      <div class="thinking-phases">
        <div class="phase-indicator active" data-phase="thinking">🤔 Thinking...</div>
        <div class="phase-indicator" data-phase="processing">⚡ Processing...</div>
        <div class="phase-indicator" data-phase="responding">💭 Responding...</div>
      </div>
    `;

    this.messageArea.appendChild(this.typingIndicatorElement);
    this.scrollToBottom();

    // Animate through phases
    setTimeout(() => this.updateThinkingPhase('processing'), 1000);
    setTimeout(() => this.updateThinkingPhase('responding'), 2000);
  }

  private updateThinkingPhase(phase: 'thinking' | 'processing' | 'responding'): void {
    if (!this.typingIndicatorElement) return;

    const indicators = this.typingIndicatorElement.querySelectorAll('.phase-indicator');
    indicators.forEach(indicator => {
      indicator.classList.remove('active');
      if (indicator.getAttribute('data-phase') === phase) {
        indicator.classList.add('active');
      }
    });
  }

  private removeTypingIndicator(): void {
    if (this.typingIndicatorElement) {
      this.typingIndicatorElement.remove();
      this.typingIndicatorElement = null;
    }
  }

  private async displayMessage(text: string, sender: 'user' | 'aura' | 'error'): Promise<void> {
    this.removeTypingIndicator();

    const messageBubble = document.createElement('div');
    messageBubble.className = `message-bubble ${sender}`;
    messageBubble.setAttribute('role', 'log');
    messageBubble.setAttribute('data-timestamp', Date.now().toString());

    try {
      messageBubble.innerHTML = await marked.parse(text);
    } catch (error) {
      console.warn('Markdown parsing failed, using plain text:', error);
      messageBubble.textContent = text;
    }

    // Always append new messages to ensure proper chronological order
    this.messageArea.appendChild(messageBubble);


    this.scrollToBottom();
  }

  private scrollToBottom(): void {
    // Scroll to bottom for normal layout with smooth behavior
    if (this.messageArea) {
      setTimeout(() => {
        this.messageArea.scrollTo({
          top: this.messageArea.scrollHeight,
          behavior: 'smooth'
        });
      }, 50); // Small delay to ensure content is rendered
    }
  }

  private setFormState(disabled: boolean): void {
    this.messageInput.disabled = disabled;
    this.sendButton.disabled = disabled;

    if (!disabled) {
      this.messageInput.focus();
    }
  }

  private updateEmotionalState(emotionalState: EmotionalState): void {
    try {
      this.emotionStatusElement.textContent = emotionalState.name;
      this.emotionDetailsElement.textContent = `Brainwave: ${emotionalState.brainwave} | NT: ${emotionalState.neurotransmitter}`;

      console.log(`🎭 Updated emotion: ${emotionalState.name} (${emotionalState.intensity})`);
    } catch (error) {
      console.warn('Failed to update emotional state display:', error);
    }
  }

  private updateCognitiveState(cognitiveState: CognitiveState): void {
    try {
      this.cognitiveFocusElement.textContent = cognitiveState.focus;
      this.cognitiveFocusDetailsElement.textContent = cognitiveState.description;

      console.log(`🧠 Updated cognitive focus: ${cognitiveState.focus}`);
    } catch (error) {
      console.warn('Failed to update cognitive state display:', error);
    }
  }

  // ============================================================================
  // CHAT HISTORY MANAGEMENT
  // ============================================================================

  private async loadChatHistory(): Promise<void> {
    if (!this.userName || !this.backendConnected) return;

    try {
      console.log("📚 Loading chat history...");
      const historyData = await this.api.getChatHistory(this.userName, 20);

      console.log("📊 Chat history response:", historyData);

      if (historyData && historyData.sessions && historyData.sessions.length > 0) {
        this.chatSessions = historyData.sessions;
        this.renderChatHistory();
        console.log(`✅ Loaded ${this.chatSessions.length} chat sessions`);
      } else if (historyData && (historyData as any).error) {
        console.error("🚨 Database error in chat history:", (historyData as any).error);
        this.renderDatabaseError();
      } else {
        console.log("📭 No chat history found");
        this.renderNoChatHistory();
      }
    } catch (error) {
      console.error("❌ Failed to load chat history:", error);
      this.renderChatHistoryError();
    }
  }

  private renderChatHistory(): void {
    this.chatHistoryList.innerHTML = this.chatSessions.map(session => {
      const isActive = session.session_id === this.currentSessionId;
      const timestamp = this.formatTimestamp(session.timestamp);
      const preview = session.last_message ?
        session.last_message.substring(0, 50) + '...' :
        'New conversation';

      return `
        <div class="chat-session-item ${isActive ? 'active' : ''}"
             data-session-id="${session.session_id}"
             role="button"
             tabindex="0">
          <div class="session-title">${this.escapeHtml(preview)}</div>
          <div class="session-meta">${timestamp} • ${session.message_count || 0} messages</div>
        </div>
      `;
    }).join('');

    // Add click handlers
    this.chatHistoryList.querySelectorAll('.chat-session-item').forEach(item => {
      item.addEventListener('click', () => {
        const sessionId = item.getAttribute('data-session-id');
        if (sessionId && sessionId !== this.currentSessionId) {
          this.loadChatSession(sessionId);
        }
      });

      // Keyboard accessibility
      item.addEventListener('keydown', (e: Event) => {
        const keyboardEvent = e as KeyboardEvent;
        if (keyboardEvent.key === 'Enter' || keyboardEvent.key === ' ') {
          e.preventDefault();
          (item as HTMLElement).click();
        }
      });
    });
  }

  private renderNoChatHistory(): void {
    this.chatHistoryList.innerHTML = '<div class="no-history">No conversation history yet. Start a new chat!</div>';
  }

  private renderChatHistoryError(): void {
    this.chatHistoryList.innerHTML = '<div class="error-message">Failed to load chat history</div>';
  }

  private renderDatabaseError(): void {
    this.chatHistoryList.innerHTML = `
      <div class="database-error">
        <div class="error-icon">🚨</div>
        <div class="error-title">Database Connection Issue</div>
        <div class="error-message">
          The database is experiencing I/O errors. This is usually due to:
          <ul>
            <li>Multiple ChromaDB instances running</li>
            <li>Disk space or permission issues</li>
            <li>Corrupted database files</li>
          </ul>
        </div>
        <button onclick="window.location.reload()" class="retry-button">
          Restart Application
        </button>
      </div>
    `;
  }

  private async loadChatSession(sessionId: string): Promise<void> {
    try {
      console.log(`📖 Loading session: ${sessionId}`);

      // Update current session
      this.currentSessionId = sessionId;

      // Update active session UI
      this.updateActiveSession(sessionId);

      // Load and display all messages for the selected session
      const sessionMessages = await this.api.getSessionMessages(this.userName!, sessionId);

      // Clear current chat area
      this.clearChat();

      if (sessionMessages && Array.isArray(sessionMessages) && sessionMessages.length > 0) {
        // Display each message in order
        for (const msg of sessionMessages) {
          // Assume msg has { sender: 'user' | 'aura', content: string }
          // Fallback to 'aura' if sender is missing
          await this.displayMessage(msg.content, msg.sender === 'user' ? 'user' : 'aura');
        }
      } else {
        await this.displayMessage('No messages found for this session.', 'aura');
      }

      // Find selected session data for display
      const selectedSessionData = this.chatSessions.find(session => session.session_id === sessionId);
      if (selectedSessionData) {
        await this.displayMessage(
          `Session: ${sessionId}\nTotal Messages: ${selectedSessionData.message_count}\nLast message: "${selectedSessionData.last_message}"\n(To view the full conversation, this section needs to be updated to fetch and display all ${selectedSessionData.message_count} messages.)`,
          'aura'
        );
      }

      console.log(`✅ Loaded session: ${sessionId}`);

    } catch (error) {
      console.error(`❌ Failed to load session ${sessionId}:`, error);
      await this.displayMessage("Failed to load this conversation. Please try again.", 'error');
    }
  }

  private async createNewChat(): Promise<void> {
    try {
      console.log("🔄 Creating new chat session...");

      // Generate new session ID
      this.currentSessionId = this.generateSessionId();

      // Clear current chat
      this.clearChat();

      // Start fresh conversation
      await this.startChat();

      // Refresh chat history when a new chat is created
      await this.loadChatHistory();

      console.log(`✅ New chat session created: ${this.currentSessionId}`);

    } catch (error) {
      console.error("❌ Failed to create new chat:", error);
      await this.displayMessage("Error creating new chat. Please refresh the page.", 'error');
    }
  }

  private clearChat(): void {
    this.messageArea.innerHTML = '';
    this.removeTypingIndicator();
  }

  private updateActiveSession(sessionId: string): void {
    this.chatHistoryList.querySelectorAll('.chat-session-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-session-id') === sessionId);
    });
  }

  private generateSessionId(): string {
    // Use crypto.randomUUID if available (modern browsers), fallback to a manual UUID v4 generator
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return `session_${crypto.randomUUID()}`;
    }
    // Fallback: manual UUID v4 generator
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
    return `session_${uuid}`;
  }

  // ============================================================================
  // MEMORY & INSIGHTS
  // ============================================================================

  private displaySearchResults(response: any, resultsElement: HTMLElement): void {
    if (response.results && response.results.length > 0) {
      resultsElement.innerHTML = response.results.map((result: any) => {
        const content = result.content.replace(/[*#]/g, '').trim();
        const similarity = (result.similarity * 100).toFixed(1);
        const source = response.includes_video_archives ? 'unified' : 'active';

        return `
          <div class="memory-result" data-source="${source}">
            <div class="memory-content">${this.escapeHtml(content)}</div>
            <div class="memory-meta">
              Relevance: ${similarity}% | Source: ${response.search_type || 'Memory'}
            </div>
          </div>
        `;
      }).join('');

      resultsElement.innerHTML += `
        <div class="search-summary">
          <strong>Search Results:</strong> ${response.results.length} found
          ${response.includes_video_archives ? ' (includes video archives)' : ''}
        </div>
      `;
    } else {
      resultsElement.innerHTML = `
        <div class="memory-result">
          <div class="memory-content">No results found</div>
          <div class="memory-meta">Try different keywords or check your search terms</div>
        </div>
      `;
    }
  }

  private displayEmotionalInsights(analysis: any, insightsElement: HTMLElement): void {
    if (analysis.dominant_emotions && analysis.dominant_emotions.length > 0) {
      const stability = (analysis.emotional_stability * 100).toFixed(1);
      const dominantEmotion = analysis.dominant_emotions[0][0];

      insightsElement.innerHTML = `
        <div class="insights-data">
          <p><strong>🎭 Dominant Emotion:</strong> ${dominantEmotion}</p>
          <p><strong>📊 Emotional Stability:</strong> ${stability}%</p>
          <p><strong>📈 Total Entries:</strong> ${analysis.total_entries}</p>

          <div class="emotional-breakdown">
            <h4>Emotional Distribution:</h4>
            ${analysis.dominant_emotions.slice(0, 5).map(([emotion, count]: [string, number]) => `
              <div class="emotion-stat">
                <span>${emotion}:</span>
                <span>${count} occurrences</span>
              </div>
            `).join('')}
          </div>

          ${analysis.recommendations && analysis.recommendations.length > 0 ? `
            <div class="recommendations">
              <h4>💡 Insights:</h4>
              ${analysis.recommendations.map((rec: string) => `<p>• ${this.escapeHtml(rec)}</p>`).join('')}
            </div>
          ` : ''}
        </div>
      `;
    } else {
      insightsElement.innerHTML = `
        <div class="insights-data">
          <p>No emotional data available for the selected period.</p>
          <p>Keep chatting with Aura to build up your emotional profile!</p>
        </div>
      `;
    }
  }

  private async updateVideoArchiveStatus(): Promise<void> {
    const statusElement = document.getElementById('video-archive-status');
    if (!statusElement) return;

    try {
      const response = await fetch('http://localhost:8000/memvid/status');
      const data = await response.json();

      if (data.status === 'operational') {
        statusElement.innerHTML = `
          <div class="archive-info">
            <div><strong>📦 Total Archives:</strong> ${data.archives_count}</div>
            <div><strong>🎥 Status:</strong> Operational</div>
            ${data.archives_count > 0 ? `
              <div><strong>📊 Recent Archives:</strong></div>
              <ul style="margin: 4px 0; padding-left: 20px; font-size: 0.8rem;">
                ${data.archives.slice(0, 3).map((archive: any) =>
                  `<li>${typeof archive === 'string' ? archive : archive.name || 'Unnamed Archive'}</li>`
                ).join('')}
              </ul>
            ` : '<div style="color: var(--text-secondary); font-style: italic;">No archives yet</div>'}
          </div>
        `;
      } else {
        statusElement.innerHTML = '<div class="archive-error">⚠️ Memvid service not available</div>';
      }
    } catch (error) {
      console.error('Failed to fetch video archive status:', error);
      statusElement.innerHTML = '<div class="archive-error">❌ Error loading archive status</div>';
    }
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private formatTimestamp(timestamp: string): string {
    try {
      const date = new Date(timestamp);
      if (isNaN(date.getTime())) return 'Unknown time';

      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffHours = diffMs / (1000 * 60 * 60);
      const diffDays = diffMs / (1000 * 60 * 60 * 24);

      if (diffHours < 1) return 'Just now';
      if (diffHours < 24) return `${Math.floor(diffHours)}h ago`;
      if (diffDays < 7) return `${Math.floor(diffDays)}d ago`;

      return date.toLocaleDateString();
    } catch {
      return 'Unknown time';
    }
  }

  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  private showConnectionWarning(): void {
    const warningElement = document.createElement('div');
    warningElement.className = 'connection-warning';
    warningElement.innerHTML = `
      <div style="background: var(--accent-warning); color: white; padding: 8px 16px; text-align: center; font-size: 0.9rem;">
        ⚠️ Backend connection failed. Some features may not work properly.
      </div>
    `;
    document.body.insertAdjacentElement('afterbegin', warningElement);
  }

  private showCriticalError(message: string): void {
    const errorElement = document.createElement('div');
    errorElement.className = 'critical-error';
    errorElement.innerHTML = `
      <div style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: var(--accent-error); color: white; padding: 20px; border-radius: 8px; text-align: center; z-index: 1000; max-width: 90%; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <h3 style="margin: 0 0 12px 0;">Critical Error</h3>
        <p style="margin: 0 0 16px 0;">${this.escapeHtml(message)}</p>
        <button onclick="window.location.reload()" style="background: white; color: var(--accent-error); border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: 500;">
          Refresh Page
        </button>
      </div>
    `;
    document.body.appendChild(errorElement);
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Global instance
let auraUI: AuraUIManager;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
  try {
    auraUI = new AuraUIManager();
    await auraUI.initialize();
  } catch (error) {
    console.error('Critical initialization error:', error);
  }
});

// Export for testing and debugging
export { AuraUIManager };
