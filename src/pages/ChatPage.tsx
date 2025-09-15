import { useState, useEffect } from "react";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ChatSection } from "@/components/chat/ChatSection";
import { InsightsPanel } from "@/components/chat/InsightsPanel";
import { UploadModal } from "@/components/chat/UploadModal";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

const ChatPage = () => {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [activeDocument, setActiveDocument] = useState<string | null>(null);

  // Show upload modal on initial load if no files
  useEffect(() => {
    if (uploadedFiles.length === 0) {
      setShowUploadModal(true);
    }
  }, []);

  const handleFilesUploaded = (files: File[]) => {
    const newFiles: UploadedFile[] = files.map(file => ({
      id: Math.random().toString(36).substring(7),
      name: file.name,
      size: file.size,
      type: file.type,
      uploadedAt: new Date()
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
    setShowUploadModal(false);
    
    // Auto-select first file if none selected
    if (selectedFiles.length === 0 && newFiles.length > 0) {
      setSelectedFiles([newFiles[0].id]);
      setActiveDocument(newFiles[0].name);
    }
  };

  const handleFileSelect = (fileId: string, selected: boolean) => {
    if (selected) {
      setSelectedFiles(prev => [...prev, fileId]);
    } else {
      setSelectedFiles(prev => prev.filter(id => id !== fileId));
    }
  };

  const handleFileClick = (file: UploadedFile) => {
    setActiveDocument(file.name);
    if (!selectedFiles.includes(file.id)) {
      setSelectedFiles(prev => [...prev, file.id]);
    }
  };

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
        />
        
        {/* Insights Panel */}
        <InsightsPanel 
          activeDocument={activeDocument}
          hasDocuments={uploadedFiles.length > 0}
        />
      </div>

      {/* Upload Modal */}
      <UploadModal 
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onFilesUploaded={handleFilesUploaded}
        currentFileCount={uploadedFiles.length}
      />
    </div>
  );
};

export default ChatPage;