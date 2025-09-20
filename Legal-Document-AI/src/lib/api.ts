// src/lib/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Types for better type safety
export interface SignedUrlResponse {
  file_id: string;
  signed_url: string;                 // V4 signed URL for PUT
  expires_at: string;
  upload_fields: { key: string; "Content-Type": string };
  gcs_path: string;                   // e.g. "gs://<bucket>/uploads/<file_id>/<filename>"
}


export interface UploadNotificationResponse {
  success: boolean;
  file_id: string;
  status: string;
  processing_started: boolean;
  estimated_processing_time?: string;
}

export interface FileMetadata {
  file_id: string;
  filename: string;
  file_size: number;
  content_type: string;
  uploaded_at: string;
  processing_status: string;
  processed_at?: string;
  gcs_path: string;
  metadata: Record<string, any>;
  analysis: Record<string, any>;
  insights: {
    summary_available: boolean;
    audio_available: boolean;
    report_available: boolean;
  };
  job_id?: string;
  error?: string;
}

export interface JobStatus {
  job_id: string;
  file_id: string;
  type: string;
  status: string;
  progress: number;
  started_at?: string;
  completed_at?: string;
  estimated_completion?: string;
  steps: Array<{
    name: string;
    status: string;
    progress: number;
  }>;
  error?: string;
  result?: any;
}

export interface InsightResponse {
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
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  response: string;
  confidence: number;
  sources: Array<Record<string, any>>;
  suggestions: string[];
  language?: string;
  response_time?: number;
  context_used?: boolean;
  warnings?: string[];
}

async function putToSignedUrl(signedUrl: string, file: File, contentType: string) {
  const res = await fetch(signedUrl, {
    method: "PUT",
    headers: { "Content-Type": contentType },
    body: file,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`PUT to GCS signed URL failed: ${res.status} ${text}`);
  }
}


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
    
    console.log('API Request:', {
      url,
      method: options.method || 'GET',
      hasBody: !!options.body
    });

    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      console.log('API Response:', {
        status: response.status,
        statusText: response.statusText,
        url: response.url
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorData.detail || errorMessage;
        } catch {
          // If we can't parse JSON error, use the default message
        }
        
        throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log('API Success:', { endpoint, dataKeys: Object.keys(data) });
      return data;

    } catch (error) {
      console.error('API Error:', {
        endpoint,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  // Upload Management
  async getSignedUrl(
    filename: string, 
    contentType: string, 
    fileSize?: number
  ): Promise<SignedUrlResponse> {
    return this.request<SignedUrlResponse>('/api/uploads/get-signed-url', {
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
  }): Promise<UploadNotificationResponse> {
    return this.request<UploadNotificationResponse>('/api/uploads/notify-uploaded', {
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

    const query = queryParams.toString();
    const endpoint = `/api/uploads${query ? `?${query}` : ''}`;

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
      filters_applied?: {
        status?: string;
        file_type?: string;
      };
    }>(endpoint);
  }

  async getFileMetadata(fileId: string): Promise<FileMetadata> {
    return this.request<FileMetadata>(`/api/uploads/${fileId}`);
  }

  async getFileDownloadUrl(fileId: string) {
    return this.request<{
      file_id: string;
      download_url: string;
      expires_at: string;
      filename: string;
      file_size: number;
      content_type: string;
    }>(`/api/uploads/${fileId}/download-url`);
  }

  // Processing
  async startProcessing(data: {
    file_id: string;
    processing_type?: string;
    options?: Record<string, any>;
  }) {
    return this.request<{
      job_id: string;
      file_id: string;
      status: string;
      processing_type: string;
      estimated_duration: string;
    }>('/api/process/start', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getJobStatus(jobId: string): Promise<JobStatus> {
    return this.request<JobStatus>(`/api/jobs/${jobId}`);
  }

  // Insights Generation
  async generateSummary(fileId: string, options?: Record<string, any>) {
    return this.request<{
      summary_id: string;
      file_id: string;
      status: string;
      created_at: string;
      summary_text: string;
      key_points: string[];
      summary_url: string;
      word_count: number;
      confidence_score: number;
    }>(`/api/insights/${fileId}/summary`, {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        type: 'summary',
        options: options || {},
      }),
    });
  }

  async generateAudio(fileId: string, options?: {
    voice_type?: string;
    language?: string;
    speed?: number;
  }) {
    return this.request<{
      audio_id: string;
      file_id: string;
      status: string;
      created_at: string;
      audio_url: string;
      duration: string;
      duration_seconds: number;
      voice_type: string;
      language: string;
      speed: number;
      transcript: string;
      file_size: number;
    }>(`/api/insights/${fileId}/audio`, {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        type: 'audio',
        options: options || {},
      }),
    });
  }

  async generateReport(fileId: string, options?: {
    type?: string;
    format?: string;
    language?: string;
    include_charts?: boolean;
  }) {
    return this.request<{
      report_id: string;
      file_id: string;
      status: string;
      created_at: string;
      report_url: string;
      report_type: string;
      format: string;
      page_count: number;
      file_size: number;
      sections: string[];
    }>(`/api/insights/${fileId}/report`, {
      method: 'POST',
      body: JSON.stringify({
        file_id: fileId,
        type: 'report',
        options: options || {},
      }),
    });
  }

  async getInsights(fileId: string): Promise<InsightResponse> {
    return this.request<InsightResponse>(`/api/insights/${fileId}`);
  }

  // Chat
  async sendMessage(data: {
    message: string;
    file_id?: string;
    conversation_id?: string;
    stream?: boolean;
  }): Promise<ChatResponse> {
    return this.request<ChatResponse>('/api/chat', {
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

  // Health and utility
  async healthCheck() {
    return this.request<{
      status: string;
      service: string;
      version: string;
      timestamp: string;
      services: {
        loaded: number;
        available: string[];
      };
      storage: {
        files: number;
        jobs: number;
        conversations: number;
      };
    }>('/health');
  }

  // Download URLs - construct the full URL for downloads
  getDownloadUrl(endpoint: string): string {
    if (endpoint.startsWith('http')) {
      return endpoint;
    }
    return `${this.baseUrl}${endpoint}`;
  }

  // File upload helper - handles the full upload flow
  async uploadFile(file: File): Promise<{
  file_id: string;
  processing_started: boolean;
  job_id?: string;
}> {
  try {
    console.log("Starting file upload:", { name: file.name, size: file.size, type: file.type });

    // 1) Ask backend for a signed URL
    const signed = await this.getSignedUrl(file.name, file.type || "application/pdf", file.size);
    console.log("Signed URL obtained:", signed.file_id);

    // 2) Upload bytes to GCS (critical!)
    await putToSignedUrl(signed.signed_url, file, file.type || "application/pdf");

    // 3) Compute gcs_path to notify the backend
    // Prefer the backend-provided value. If missing, build one using your bucket name env.
    let gcs_path = signed.gcs_path;
    if (!gcs_path) {
      const bucket = import.meta.env.VITE_GCS_BUCKET_NAME; // <-- add this to your .env for the frontend
      if (!bucket) {
        throw new Error("Missing gcs_path in get-signed-url response and VITE_GCS_BUCKET_NAME is not set");
      }
      gcs_path = `gs://${bucket}/${signed.upload_fields.key}`;
    }

    // 4) Tell backend the object is in GCS
    const notification = await this.notifyUploaded({
      file_id: signed.file_id,
      gcs_path,
      original_filename: file.name,
      file_size: file.size,
      content_type: file.type || "application/pdf",
    });

    console.log("Upload notification sent:", notification);

    return {
      file_id: signed.file_id,
      processing_started: notification.processing_started,
    };
  } catch (error) {
    console.error("Upload process failed:", error);
    throw error;
  }
}

}

export const apiService = new ApiService();
export default apiService;