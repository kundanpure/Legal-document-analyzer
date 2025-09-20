// src/hooks/api.ts
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService, { 
  FileMetadata, 
  JobStatus, 
  InsightResponse,
  ChatResponse,
} from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

// Query key factory for consistent cache management
const queryKeys = {
  files: (params?: any) => ['files', params] as const,
  fileMetadata: (fileId: string) => ['file-metadata', fileId] as const,
  jobStatus: (jobId: string) => ['job-status', jobId] as const,
  insights: (fileId: string) => ['insights', fileId] as const,
  health: () => ['health'] as const,
};

// Upload hooks
export const useFiles = (params?: {
  limit?: number;
  offset?: number;
  status?: string;
  file_type?: string;
}) => {
  return useQuery({
    queryKey: queryKeys.files(params),
    queryFn: () => apiService.listFiles(params),
    staleTime: 30000, // 30 seconds
    retry: 2,
    refetchOnWindowFocus: false,
  });
};

export const useFileMetadata = (fileId: string | null) => {
  const queryClient = useQueryClient();
  
  const query = useQuery({
    queryKey: queryKeys.fileMetadata(fileId!),
    queryFn: () => apiService.getFileMetadata(fileId!),
    enabled: !!fileId,
    refetchIntervalInBackground: false,
    retry: (failureCount, error) => {
      // Don't retry if file not found
      if (error instanceof Error && error.message.includes('404')) {
        return false;
      }
      return failureCount < 3;
    },
  });

  // Handle auto-refetching for processing status
  React.useEffect(() => {
    if (!query.data || !fileId) return;
    
    const isProcessing = query.data.processing_status === 'processing';
    
    if (isProcessing) {
      const interval = setInterval(() => {
        query.refetch();
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [query.data?.processing_status, query.refetch, fileId]);

  // Invalidate insights when processing completes
  React.useEffect(() => {
    if (query.data?.processing_status === 'completed' && fileId) {
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.insights(fileId) 
      });
    }
  }, [query.data?.processing_status, fileId, queryClient]);

  return query;
};

export const useUploadFile = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (file: File): Promise<{
      file_id: string;
      processing_started: boolean;
      job_id?: string;
    }> => {
      return await apiService.uploadFile(file);
    },
    onSuccess: (data) => {
      // Invalidate files list to show new file
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.files() 
      });
      
      toast({
        title: 'Upload successful',
        description: `File uploaded and processing ${data.processing_started ? 'started' : 'queued'}`,
      });

      // Start tracking the file metadata
      queryClient.prefetchQuery({
        queryKey: queryKeys.fileMetadata(data.file_id),
        queryFn: () => apiService.getFileMetadata(data.file_id),
      });
    },
    onError: (error: Error) => {
      console.error('Upload error:', error);
      toast({
        title: 'Upload failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

// Processing hooks
export const useJobStatus = (jobId: string | null) => {
  const query = useQuery({
    queryKey: queryKeys.jobStatus(jobId!),
    queryFn: () => apiService.getJobStatus(jobId!),
    enabled: !!jobId,
    refetchIntervalInBackground: false,
    retry: 2,
  });

  // Handle auto-refetching for processing status
  React.useEffect(() => {
    if (!query.data || !jobId) return;
    
    const isProcessing = query.data.status === 'processing' || query.data.status === 'queued';
    
    if (isProcessing) {
      const interval = setInterval(() => {
        query.refetch();
      }, 3000);
      
      return () => clearInterval(interval);
    }
  }, [query.data?.status, query.refetch, jobId]);

  return query;
};

// Insights hooks
export const useInsights = (fileId: string | null) => {
  return useQuery({
    queryKey: queryKeys.insights(fileId!),
    queryFn: () => apiService.getInsights(fileId!),
    enabled: !!fileId,
    staleTime: 60000, // 1 minute
    retry: 2,
  });
};

export const useGenerateSummary = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ 
      fileId, 
      options 
    }: { 
      fileId: string; 
      options?: Record<string, any> 
    }) => apiService.generateSummary(fileId, options),
    onSuccess: (data) => {
      // Update insights cache
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.insights(data.file_id) 
      });
      
      // Update file metadata cache  
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.fileMetadata(data.file_id) 
      });

      toast({
        title: 'Summary generated',
        description: 'Document summary is ready for download',
      });
    },
    onError: (error: Error) => {
      console.error('Summary generation error:', error);
      toast({
        title: 'Summary generation failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

export const useGenerateAudio = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ 
      fileId, 
      options 
    }: { 
      fileId: string; 
      options?: { 
        voice_type?: string; 
        language?: string; 
        speed?: number; 
      } 
    }) => apiService.generateAudio(fileId, options),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.insights(data.file_id) 
      });
      
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.fileMetadata(data.file_id) 
      });

      toast({
        title: 'Audio generated',
        description: `Audio summary (${data.duration}) is ready for download`,
      });
    },
    onError: (error: Error) => {
      console.error('Audio generation error:', error);
      toast({
        title: 'Audio generation failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

export const useGenerateReport = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ 
      fileId, 
      options 
    }: { 
      fileId: string; 
      options?: { 
        type?: string; 
        format?: string; 
        language?: string; 
        include_charts?: boolean;
      } 
    }) => apiService.generateReport(fileId, options),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.insights(data.file_id) 
      });
      
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.fileMetadata(data.file_id) 
      });

      toast({
        title: 'Report generated',
        description: `${data.page_count}-page ${data.format} report is ready for download`,
      });
    },
    onError: (error: Error) => {
      console.error('Report generation error:', error);
      toast({
        title: 'Report generation failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

// Chat hooks
export const useChat = () => {
  const { toast } = useToast();

  return useMutation({
    mutationFn: (data: {
      message: string;
      file_id?: string;
      conversation_id?: string;
      stream?: boolean;
    }): Promise<ChatResponse> => apiService.sendMessage(data),
    onError: (error: Error) => {
      console.error('Chat error:', error);
      toast({
        title: 'Chat error',
        description: error.message || 'Failed to send message',
        variant: 'destructive',
      });
    },
  });
};

// Enhanced chat hook with conversation management
export const useChatConversation = (fileId?: string) => {
  const [conversationId, setConversationId] = React.useState<string | null>(null);
  const [messages, setMessages] = React.useState<Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    confidence?: number;
    sources?: any[];
  }>>([]);

  const chatMutation = useChat();

  const sendMessage = React.useCallback(async (message: string) => {
    if (!message.trim()) return;

    // Add user message immediately
    const userMessage = {
      id: Date.now().toString(),
      role: 'user' as const,
      content: message,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await chatMutation.mutateAsync({
        message,
        file_id: fileId,
        conversation_id: conversationId || undefined,
      });

      // Update conversation ID if this was the first message
      if (!conversationId) {
        setConversationId(response.conversation_id);
      }

      // Add assistant response
      const assistantMessage = {
        id: response.message_id,
        role: 'assistant' as const,
        content: response.response,
        timestamp: new Date(),
        confidence: response.confidence,
        sources: response.sources,
      };

      setMessages(prev => [...prev, assistantMessage]);

      return response;

    } catch (error) {
      // Remove the user message if sending failed
      setMessages(prev => prev.filter(msg => msg.id !== userMessage.id));
      throw error;
    }
  }, [fileId, conversationId, chatMutation]);

  const clearConversation = React.useCallback(() => {
    setMessages([]);
    setConversationId(null);
  }, []);

  return {
    messages,
    conversationId,
    sendMessage,
    clearConversation,
    isLoading: chatMutation.isPending,
    error: chatMutation.error,
  };
};

// Export hooks
export const useExportConversation = () => {
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ 
      conversationId, 
      format = 'pdf' 
    }: { 
      conversationId: string; 
      format?: string; 
    }) => apiService.exportConversation(conversationId, format),
    onSuccess: (data) => {
      toast({
        title: 'Export successful',
        description: `Conversation exported as ${data.format.toUpperCase()}`,
      });
      
      // Trigger download
      const downloadUrl = apiService.getDownloadUrl(data.download_url);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `conversation_${data.export_id}.${data.format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    },
    onError: (error: Error) => {
      console.error('Export error:', error);
      toast({
        title: 'Export failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
};

// Health check hook
export const useHealthCheck = () => {
  return useQuery({
    queryKey: queryKeys.health(),
    queryFn: () => apiService.healthCheck(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 3,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
  });
};

// Utility hook for downloading files
export const useDownloadFile = () => {
  const { toast } = useToast();

  return React.useCallback((url: string, filename?: string) => {
    try {
      const downloadUrl = apiService.getDownloadUrl(url);
      const link = document.createElement('a');
      link.href = downloadUrl;
      if (filename) {
        link.download = filename;
      }
      link.target = '_blank';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      toast({
        title: 'Download failed',
        description: error instanceof Error ? error.message : 'Failed to download file',
        variant: 'destructive',
      });
    }
  }, [toast]);
};

// Hook for tracking multiple file processing statuses
export const useFileProcessingQueue = () => {
  const [processingFiles, setProcessingFiles] = React.useState<Set<string>>(new Set());

  const addToQueue = React.useCallback((fileId: string) => {
    setProcessingFiles(prev => new Set([...prev, fileId]));
  }, []);

  const removeFromQueue = React.useCallback((fileId: string) => {
    setProcessingFiles(prev => {
      const newSet = new Set(prev);
      newSet.delete(fileId);
      return newSet;
    });
  }, []);

  const clearQueue = React.useCallback(() => {
    setProcessingFiles(new Set());
  }, []);

  return {
    processingFiles: Array.from(processingFiles),
    addToQueue,
    removeFromQueue,
    clearQueue,
    isProcessing: (fileId: string) => processingFiles.has(fileId),
    hasProcessingFiles: processingFiles.size > 0,
  };
};

// Hook for batch operations
export const useBatchOperations = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const refreshAllFiles = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['files'] });
    toast({
      title: 'Refreshing files',
      description: 'Updating file list and status...',
    });
  }, [queryClient, toast]);

  const refreshFileData = React.useCallback((fileId: string) => {
    queryClient.invalidateQueries({ queryKey: queryKeys.fileMetadata(fileId) });
    queryClient.invalidateQueries({ queryKey: queryKeys.insights(fileId) });
  }, [queryClient]);

  return {
    refreshAllFiles,
    refreshFileData,
  };
};