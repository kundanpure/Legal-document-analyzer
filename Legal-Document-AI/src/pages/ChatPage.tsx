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
  const [mobileView, setMobileView] = useState<"chat" | "sources" | "insights">("chat");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [showSourcesDesktop, setShowSourcesDesktop] = useState(true);
  const [showInsightsDesktop, setShowInsightsDesktop] = useState(true);
  const [showSourcesMobile, setShowSourcesMobile] = useState(false);
  const [showInsightsMobile, setShowInsightsMobile] = useState(false);

  const { toast } = useToast();
  const { data: filesData, isLoading, error, refetch } = useFiles();
  const uploadMutation = useUploadFile();

  const uploadedFiles: UploadedFile[] =
    filesData?.files?.map((file: any) => ({
      id: file.file_id,
      name: file.filename,
      size: file.file_size,
      type: file.content_type,
      uploadedAt: new Date(file.uploaded_at),
    })) || [];

  useEffect(() => {
    if (!isLoading && uploadedFiles.length === 0) {
      setShowUploadModal(true);
    }
  }, [isLoading, uploadedFiles.length]);

  useEffect(() => {
    if (error) {
      toast({
        title: "Error loading files",
        description: error.message,
        variant: "destructive",
      });
    }
  }, [error, toast]);

  useEffect(() => {
    if (selectedFiles.length === 0 && uploadedFiles.length > 0) {
      const firstFile = uploadedFiles[0];
      setSelectedFiles([firstFile.id]);
      setActiveDocument(firstFile.name);
      setActiveFileId(firstFile.id);
    }
  }, [uploadedFiles, selectedFiles.length]);

  const handleFilesUploaded = async (files: File[]) => {
    try {
      for (const file of files) {
        await uploadMutation.mutateAsync(file);
      }
      await refetch();
      setShowUploadModal(false);
    } catch (err) {
      console.error("Upload failed:", err);
    }
  };

  const handleFileSelect = (fileId: string, selected: boolean) => {
    if (selected) {
      setSelectedFiles((prev) => [...prev, fileId]);
    } else {
      setSelectedFiles((prev) => prev.filter((id) => id !== fileId));
    }
  };

  const handleFileClick = (file: UploadedFile) => {
    setActiveDocument(file.name);
    setActiveFileId(file.id);
    if (!selectedFiles.includes(file.id)) {
      setSelectedFiles((prev) => [...prev, file.id]);
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
    <div className="font-poppins bg-background flex flex-col h-screen">
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap');
          .font-poppins { font-family: 'Poppins', sans-serif; }
          .mobile-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 40; }
          .mobile-panel { position: fixed; top: 0; bottom: 0; width: 80%; max-width: 360px; z-index: 50; transition: transform 0.28s ease-in-out; display:flex; flex-direction:column; }
          .mobile-panel.sources { left: 0; transform: translateX(-100%); }
          .mobile-panel.sources.open { transform: translateX(0); }
          .mobile-panel.insights { right: 0; transform: translateX(100%); }
          .mobile-panel.insights.open { transform: translateX(0); }
          @media (min-width: 900px) { .mobile-only { display:none; } }
          @media (max-width: 899px) { .desktop-only { display:none; } }
        `}
      </style>

      {/* This mobile header is fixed, so it's out of the layout flow. */}
      <header className="fixed top-0 left-0 w-full z-30 bg-background py-4 px-6 flex justify-between items-center md:hidden border-b border-gray-800">
        <a href="/" className="text-xl font-bold text-foreground z-30">Legal AI</a>
        <button onClick={() => setIsMobileMenuOpen((prev) => !prev)} className="z-30">
          {isMobileMenuOpen ? (
            <svg className="w-6 h-6 text-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          ) : (
            <svg className="w-6 h-6 text-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7"></path>
            </svg>
          )}
        </button>
      </header>

      {isMobileMenuOpen && (
        <div className="md:hidden fixed inset-0 top-16 z-20 bg-background p-6 pt-8">
          <nav className="flex flex-col items-start space-y-4">
            <a href="#" onClick={(e) => e.preventDefault()} className="text-lg text-muted-foreground hover:text-foreground transition-colors">Overview</a>
            <a href="#" onClick={(e) => e.preventDefault()} className="text-lg text-muted-foreground hover:text-foreground transition-colors">Features</a>
            <a href="#" onClick={(e) => e.preventDefault()} className="text-lg text-muted-foreground hover:text-foreground transition-colors">Contact</a>
          </nav>
        </div>
      )}

      {/* MODIFICATION: 
        Added 'pt-16 md:pt-0'
        'pt-16' (4rem) offsets the content to account for the fixed mobile header.
        'md:pt-0' removes this padding on desktop, where the fixed header is hidden.
      */}
      <main className="flex-1 flex flex-row pt-16 md:pt-0" style={{ overflow: "hidden" }}>
        <div
          className={`${mobileView === "sources" ? "block w-full h-full" : "hidden"} md:block md:h-full`}
          style={{
            overflow: "hidden",
            width: showSourcesDesktop ? undefined : 0,
            transition: "width 200ms ease",
          }}
        >
          {showSourcesDesktop && (
            <div className="md:w-80 lg:w-96 h-full">
              <SourcesPanel
                uploadedFiles={uploadedFiles}
                selectedFiles={selectedFiles}
                onFileSelect={handleFileSelect}
                onFileClick={handleFileClick}
                onAddMore={() => setShowUploadModal(true)}
              />
            </div>
          )}
        </div>

        <div
          className={`${mobileView === "chat" ? "flex" : "hidden"} flex-1 flex-col`}
          style={{ minHeight: 0, overflow: "hidden", transition: "width 200ms ease" }}
        >
          <ChatSection
            activeDocument={activeDocument}
            hasDocuments={uploadedFiles.length > 0}
            activeFileId={activeFileId}
            onConversationIdChange={handleConversationIdChange}
            showSourcesDesktop={showSourcesDesktop}
            setShowSourcesDesktop={setShowSourcesDesktop}
            showInsightsDesktop={showInsightsDesktop}
            setShowInsightsDesktop={setShowInsightsDesktop}
          />
        </div>

        <div
          className={`${mobileView === "insights" ? "block w-full h-full" : "hidden"} md:hidden lg:block lg:h-full`}
          style={{
            overflow: "hidden",
            width: showInsightsDesktop ? undefined : 0,
            transition: "width 200ms ease",
          }}
        >
          {showInsightsDesktop && (
            <div className="lg:w-80 h-full">
              <InsightsPanel
                activeDocument={activeDocument}
                hasDocuments={uploadedFiles.length > 0}
                activeFileId={activeFileId}
                conversationId={conversationId}
              />
            </div>
          )}
        </div>
      </main>

      {/* This mobile footer is part of the flex-col, so it correctly
        sits at the bottom of the h-screen view. No changes needed.
      */}
      <div className="md:hidden flex justify-around p-2 border-t bg-background shadow-sm mobile-only" style={{ position: "relative", zIndex: 10 }}>
        <button
          onClick={() => {
            setMobileView("sources");
            setShowSourcesMobile(false);
            setShowInsightsMobile(false);
          }}
          className={`px-3 py-1 rounded-md text-sm font-medium ${mobileView === "sources" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
        >
          Sources
        </button>
        <button
          onClick={() => {
            setMobileView("chat");
            setShowSourcesMobile(false);
            setShowInsightsMobile(false);
          }}
          className={`px-3 py-1 rounded-md text-sm font-medium ${mobileView === "chat" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
        >
          Chat
        </button>
        <button
          onClick={() => {
            setMobileView("insights");
            setShowSourcesMobile(false);
            setShowInsightsMobile(false);
          }}
          className={`px-3 py-1 rounded-md text-sm font-medium ${mobileView === "insights" ? "bg-primary text-primary-foreground" : "text-muted-foreground"}`}
        >
          Insights
        </button>
      </div>

      {showSourcesMobile && (
        <>
          <div className="mobile-overlay" onClick={() => setShowSourcesMobile(false)} />
          <div className="mobile-panel sources open">
            <SourcesPanel
              uploadedFiles={uploadedFiles}
              selectedFiles={selectedFiles}
              onFileSelect={handleFileSelect}
              onFileClick={handleFileClick}
              onAddMore={() => setShowUploadModal(true)}
            />
          </div>
        </>
      )}

      {showInsightsMobile && (
        <>
          <div className="mobile-overlay" onClick={() => setShowInsightsMobile(false)} />
          <div className="mobile-panel insights open">
            <InsightsPanel
              activeDocument={activeDocument}
              hasDocuments={uploadedFiles.length > 0}
              activeFileId={activeFileId}
              conversationId={conversationId}
            />
          </div>
        </>
      )}

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