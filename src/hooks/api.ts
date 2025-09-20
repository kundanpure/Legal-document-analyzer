// src/hooks/api.ts
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import apiService from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

// Upload hooks
export const useFiles = (params?: {
  limit?: number;
  offset?: number;
  status?: string;
  file_type?: string;
}) => {
  return useQuery({
    queryKey: ['files', params],
    queryFn: () => apiService.listFiles(params),
    staleTime: 30000, // 30 seconds
  });
};

export const useFileMetadata = (fileId: string | null) => {
  const query = useQuery({
    queryKey: ['file-metadata', fileId],
    queryFn: () => apiService.getFileMetadata(fileId!),
    enabled: !!fileId,
  });

  // Set up conditional refetching based on processing status
  React.useEffect(() => {
    if (query.data?.processing_status === 'processing') {
      const interval = setInterval(() => {
        query.refetch();
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [query.data?.processing_status, query.refetch]);

  return query;
};

export const useUploadFile = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (file: File) => {
      // Step 1: Get signed URL
      const signedUrlData = await apiService.getSignedUrl(
        file.name,
        file.type,
        file.size
      );

      // Step 2: Upload to GCS (simplified - in reality you'd upload to the signed URL)
      const mockGcsPath = `uploads/${signedUrlData.file_id}/${file.name}`;

      // Step 3: Notify upload completion
      const notificationData = await apiService.notifyUploaded({
        file_id: signedUrlData.file_id,
        gcs_path: mockGcsPath,
        original_filename: file.name,
        file_size: file.size,
        content_type: file.type,
      });

      return { ...signedUrlData, ...notificationData };
    },
    onSuccess: (data) => {
      // Invalidate files list to refresh
      queryClient.invalidateQueries({ queryKey: ['files'] });
      
      toast({
        title: 'Upload successful',
        description: `File ${data.file_id} uploaded and processing started`,
      });
    },
    onError: (error) => {
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
    queryKey: ['job-status', jobId],
    queryFn: () => apiService.getJobStatus(jobId!),
    enabled: !!jobId,
  });

  // Set up conditional refetching based on job status
  React.useEffect(() => {
    if (query.data?.status === 'processing') {
      const interval = setInterval(() => {
        query.refetch();
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [query.data?.status, query.refetch]);

  return query;
};

// Insights hooks
export const useInsights = (fileId: string | null) => {
  return useQuery({
    queryKey: ['insights', fileId],
    queryFn: () => apiService.getInsights(fileId!),
    enabled: !!fileId,
  });
};

export const useGenerateSummary = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: ({ fileId, options }: { fileId: string; options?: any }) =>
      apiService.generateSummary(fileId, options),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['insights', data.file_id] });
      toast({
        title: 'Summary generated',
        description: 'Document summary is ready for download',
      });
    },
    onError: (error) => {
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
      queryClient.invalidateQueries({ queryKey: ['insights', data.file_id] });
      toast({
        title: 'Audio generated',
        description: 'Document audio is ready for download',
      });
    },
    onError: (error) => {
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
      } 
    }) => apiService.generateReport(fileId, options),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['insights', data.file_id] });
      toast({
        title: 'Report generated',
        description: 'Document report is ready for download',
      });
    },
    onError: (error) => {
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
    }) => apiService.sendMessage(data),
    onError: (error) => {
      toast({
        title: 'Chat error',
        description: error.message,
        variant: 'destructive',
      });
    },
  });
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
        description: 'Conversation exported successfully',
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
    onError: (error) => {
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
    queryKey: ['health'],
    queryFn: () => apiService.healthCheck(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 3,
  });
};