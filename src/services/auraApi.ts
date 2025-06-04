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

export interface SearchRequest {
  user_id: string;
  query: string;
  n_results?: number;
}

export interface SearchResponse {
  results: Array<{
    content: string;
    metadata: any;
    similarity: number;
  }>;
}

export interface EmotionalAnalysisResponse {
  period_days: number;
  period_type?: string;
  custom_days?: number;
  total_entries: number;
  dominant_emotions: Array<[string, number]>;
  intensity_distribution: Record<string, number>;
  brainwave_patterns?: Record<string, number>;
  emotional_stability: number;
  recommendations: string[];
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
    try {
      const response = await fetch(`${API_BASE_URL}/conversation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API error ${response.status}: ${errorText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error sending message to Aura backend:', error);
      throw error;
    }
  }

  async searchMemories(userId: string, query: string, nResults: number = 5): Promise<SearchResponse> {
    try {
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
        const errorText = await response.text();
        throw new Error(`Search error ${response.status}: ${errorText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error searching memories:', error);
      throw error;
    }
  }

  async getEmotionalAnalysis(userId: string, period: string = 'week', customDays?: number): Promise<EmotionalAnalysisResponse> {
    try {
      let url = `${API_BASE_URL}/emotional-analysis/${userId}?period=${period}`;
      if (customDays !== undefined) {
        url += `&custom_days=${customDays}`;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis error ${response.status}: ${errorText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error getting emotional analysis:', error);
      throw error;
    }
  }

  async getChatHistory(userId: string, limit: number = 50) {
    try {
      const response = await fetch(`${API_BASE_URL}/chat-history/${userId}?limit=${limit}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`History error ${response.status}: ${errorText}`);
      }
      
      return response.json();
    } catch (error) {
      console.error('Error getting chat history:', error);
      throw error;
    }
  }

  async exportUserData(userId: string, format: string = 'json') {
    try {
      const response = await fetch(`${API_BASE_URL}/export/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ format }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Export error ${response.status}: ${errorText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Error exporting user data:', error);
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      return response.json();
    } catch (error) {
      console.error('Error checking backend health:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const auraAPI = AuraAPI.getInstance();
