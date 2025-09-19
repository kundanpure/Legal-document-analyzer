// src/lib/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    console.log('API Request:', url, options); 

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    console.log('API Response:', response.status, response.statusText);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ 
        message: 'Network error' 
      }));
      throw new Error(error.message || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // Upload Management
  async getSignedUrl(filename: string, contentType: string, fileSize?: number) {
    return this.request<{
      file_id: string;
      signed_url: string;
      expires_at: string;
      upload_fields: Record<string, string>;
    }>('/api/uploads/get-signed-url', {
      method: 'POST',
      body: JSON.stringify({
        filename,
        content_type: contentType,
        file_size: fileSize,
      }),
    });
  }

  async notifyUploaded(data: {
    file_id: string;
    gcs_path: string;
    original_filename: string;
    file_size: number;
    content_type: string;
  }) {
    return this.request<{
      success: boolean;
      file_id: string;
      status: string;
      processing_started: boolean;
    }>('/api/uploads/notify-uploaded', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async listFiles(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    file_type?: string;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.set('limit', params.limit.toString());
    if (params?.offset) queryParams.set('offset', params.offset.toString());
    if (params?.status) queryParams.set('status', params.status);
    if (params?.file_type) queryParams.set('file_type', params.file_type);

    return this.request<{
      files: Array<{
        file_id: string;
        filename: string;
        file_size: number;
        content_type: string;
        uploaded_at: string;
        processing_status: string;
        file_type: string;
        insights_available: {
          summary: boolean;
          audio: boolean;
          report: boolean;
        };
      }>;
      pagination: {
        total: number;
        limit: number;
        offset: number;
        has_more: boolean;
      };
    }>(`/api/uploads?${queryParams}`);
  }

  async getFileMetadata(fileId: string) {
    return this.request<{
      file_id: string;
      filename: string;
      file_size: number;
      content_type: string;
      uploaded_at: string;
      processing_status: string;
      processed_at?: string;
      metadata: Record<string, any>;
      analysis: Record<string, any>;
      insights: {
        summary_available: boolean;
        audio_available: boolean;
        report_available: boolean;
      };
      job_id?: string;
      error?: string;
    }>(`/api/uploads/${fileId}`);
  }

  // Processing
  async getJobStatus(jobId: string) {
    return this.request<{
      job_id: string;
      file_id: string;
      type: string;
      status: string;
      progress: number;
      started_at?: string;
      completed_at?: string;
      steps: Array<{
        name: string;
        status: string;
        progress: number;
      }>;
      error?: string;
    }>(`/api/jobs/${jobId}`);
  }

  // Insights
  async generateSummary(fileId: string, options?: Record<string, any>) {
    return this.request<{
      file_id: string;
      summary_id: string;
      status: string;
      summary_url: string;
      created_at: string;
      word_count: number;
      confidence_score: number;
    }>(`/api/insights/${fileId}/summary`, {
      method: 'POST',
      body: JSON.stringify({ options }),
    });
  }

  async generateAudio(fileId: string, options?: {
    voice_type?: string;
    language?: string;
    speed?: number;
  }) {
    return this.request<{
      file_id: string;
      audio_id: string;
      status: string;
      audio_url: string;
      duration: string;
      voice_type: string;
      language: string;
      created_at: string;
    }>(`/api/insights/${fileId}/audio`, {
      method: 'POST',
      body: JSON.stringify({ options }),
    });
  }

  async generateReport(fileId: string, options?: {
    type?: string;
    format?: string;
    language?: string;
  }) {
    return this.request<{
      file_id: string;
      report_id: string;
      status: string;
      report_url: string;
      report_type: string;
      format: string;
      page_count: number;
      created_at: string;
    }>(`/api/insights/${fileId}/report`, {
      method: 'POST',
      body: JSON.stringify({ options }),
    });
  }

  async getInsights(fileId: string) {
    return this.request<{
      file_id: string;
      insights: {
        summary: {
          available: boolean;
          summary_id?: string;
          url?: string;
          created_at?: string;
          word_count?: number;
        };
        audio: {
          available: boolean;
          audio_id?: string;
          url?: string;
          created_at?: string;
          duration?: string;
        };
        report: {
          available: boolean;
          report_id?: string;
          url?: string;
          created_at?: string;
          page_count?: number;
        };
      };
    }>(`/api/insights/${fileId}`);
  }

  // Chat
  async sendMessage(data: {
    message: string;
    file_id?: string;
    conversation_id?: string;
  }) {
    return this.request<{
      conversation_id: string;
      message_id: string;
      response: string;
      confidence: number;
      sources: Array<Record<string, any>>;
      suggestions: string[];
    }>('/api/chat', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // Export
  async exportConversation(conversationId: string, format: string = 'pdf') {
    return this.request<{
      export_id: string;
      conversation_id: string;
      format: string;
      file_size: number;
      download_url: string;
      created_at: string;
    }>('/api/export/conversation', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: conversationId,
        format,
      }),
    });
  }

  // Health
  async healthCheck() {
    return this.request<{
      status: string;
      service: string;
      version: string;
      timestamp: string;
    }>('/health');
  }

  // Download URLs
  getDownloadUrl(endpoint: string): string {
    return `${this.baseUrl}${endpoint}`;
  }
}

export const apiService = new ApiService();
export default apiService;