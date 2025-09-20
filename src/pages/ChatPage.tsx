import { useState, useEffect } from "react";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ChatSection } from "@/components/chat/ChatSection";
import { InsightsPanel } from "@/components/chat/InsightsPanel";
import { UploadModal } from "@/components/chat/UploadModal";
import { useFiles, useUploadFile } from "@/hooks/api";
import { useToast } from "@/hooks/use-toast";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

const ChatPage = () => {
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [activeDocument, setActiveDocument] = useState<string | null>(null);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const { toast } = useToast();

  // Use the API hook to fetch files
  const { data: filesData, isLoading, error, refetch } = useFiles();
  const uploadMutation = useUploadFile();

  // Convert API data to local format
  const uploadedFiles: UploadedFile[] = filesData?.files?.map(file => ({
    id: file.file_id,
    name: file.filename,
    size: file.file_size,
    type: file.content_type,
    uploadedAt: new Date(file.uploaded_at)
  })) || [];

  // Show upload modal on initial load if no files
  useEffect(() => {
    if (!isLoading && uploadedFiles.length === 0) {
      setShowUploadModal(true);
    }
  }, [isLoading, uploadedFiles.length]);

  // Handle API errors
  useEffect(() => {
    if (error) {
      toast({
        title: "Error loading files",
        description: error.message,
        variant: "destructive",
      });
    }
  }, [error, toast]);

  const handleFilesUploaded = async (files: File[]) => {
    try {
      // Upload files one by one
      for (const file of files) {
        await uploadMutation.mutateAsync(file);
      }
      
      // Refetch the files list to get updated data
      await refetch();
      
      setShowUploadModal(false);
      
    } catch (error) {
      console.error("Upload failed:", error);
      // Error is already handled by the mutation's onError callback
    }
  };

  // Auto-select first file when files are loaded
  useEffect(() => {
    if (selectedFiles.length === 0 && uploadedFiles.length > 0) {
      const firstFile = uploadedFiles[0];
      setSelectedFiles([firstFile.id]);
      setActiveDocument(firstFile.name);
      setActiveFileId(firstFile.id);
    }
  }, [uploadedFiles, selectedFiles.length]);

  const handleFileSelect = (fileId: string, selected: boolean) => {
    if (selected) {
      setSelectedFiles(prev => [...prev, fileId]);
    } else {
      setSelectedFiles(prev => prev.filter(id => id !== fileId));
    }
  };

  const handleFileClick = (file: UploadedFile) => {
    // Set as active document
    setActiveDocument(file.name);
    setActiveFileId(file.id);
    
    // Auto-select if not already selected
    if (!selectedFiles.includes(file.id)) {
      setSelectedFiles(prev => [...prev, file.id]);
    }
  };

  const handleConversationIdChange = (newConversationId: string) => {
    setConversationId(newConversationId);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
          <p className="mt-4">Loading your documents...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="h-screen flex">
        {/* Sources Panel */}
        <SourcesPanel 
          uploadedFiles={uploadedFiles}
          selectedFiles={selectedFiles}
          onFileSelect={handleFileSelect}
          onFileClick={handleFileClick}
          onAddMore={() => setShowUploadModal(true)}
        />
        
        {/* Chat Section */}
        <ChatSection 
          activeDocument={activeDocument}
          hasDocuments={uploadedFiles.length > 0}
          activeFileId={activeFileId}
          onConversationIdChange={handleConversationIdChange}
        />
        
        {/* Insights Panel */}
        <InsightsPanel 
          activeDocument={activeDocument}
          hasDocuments={uploadedFiles.length > 0}
          activeFileId={activeFileId}
          conversationId={conversationId}
        />
      </div>

      {/* Upload Modal */}
      <UploadModal 
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onFilesUploaded={handleFilesUploaded}
        currentFileCount={uploadedFiles.length}
        isUploading={uploadMutation.isPending}
      />
    </div>
  );
};

export default ChatPage;