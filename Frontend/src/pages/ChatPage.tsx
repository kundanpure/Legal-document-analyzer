import { useState, useEffect } from "react";
import { useNavigate, useLocation, useParams, useSearchParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ChatSection } from "@/components/chat/ChatSection";
import { InsightsPanel } from "@/components/chat/InsightsPanel";
import { UploadModal } from "@/components/chat/UploadModal";
import { useFiles, useUploadFile, useChatStatus } from "@/hooks/api";
import { useToast } from "@/hooks/use-toast";
import { Clock, Server, Zap, CheckCircle, AlertCircle, FileText, User as UserIcon, LogOut, ArrowLeft } from "lucide-react";
import { logout } from "@/lib/firebase";
import { toast as sonnerToast } from "sonner";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

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
  const [processingFiles, setProcessingFiles] = useState<string[]>([]);

  const { toast } = useToast();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { chatId } = useParams<{ chatId: string }>();
  const [searchParams] = useSearchParams();

  const handleLogout = async () => {
    try {
      await logout();
      navigate("/");
      sonnerToast.success("Logged out successfully");
    } catch (error) {
      sonnerToast.error("Error logging out");
    }
  };

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

  const { data: filesData, isLoading, error, refetch } = useFiles({ chat_id: chatId });
  const uploadMutation = useUploadFile();
  const { data: statusData } = useChatStatus(chatId || null);

  useEffect(() => {
    if (statusData?.all_processed && processingFiles.length > 0) {
      sonnerToast.success("All documents processed! Ready to chat 🎉");
      setProcessingFiles([]);
      refetch(); // Refresh to get the AI generated title/summary
    } else if (statusData && !statusData.all_processed) {
      const currentlyProcessing = statusData.files.filter(f => f.processing_status === 'pending' || f.processing_status === 'processing').map(f => f.file_id);
      if (currentlyProcessing.length > 0 && processingFiles.length === 0) {
        sonnerToast("Processing documents... This usually takes ~20 seconds.");
        setProcessingFiles(currentlyProcessing);
      }
    }
  }, [statusData, processingFiles.length, refetch]);

  useEffect(() => {
    if (location.state?.activeFileId) {
      setActiveFileId(location.state.activeFileId);
      if (!selectedFiles.includes(location.state.activeFileId)) {
        setSelectedFiles(prev => [...prev, location.state.activeFileId]);
      }
    }
  }, [location.state]);

  const uploadedFiles: UploadedFile[] =
    filesData?.files?.map((file: any) => ({
      id: file.file_id,
      name: file.filename,
      size: file.file_size,
      type: file.content_type,
      uploadedAt: new Date(file.uploaded_at),
      status: file.processing_status
    })) || [];

  useEffect(() => {
    if (!isLoading && uploadedFiles.length === 0 && !searchParams.get('new')) {
      // It's empty, but not marked new? Just show the modal to be safe
      setShowUploadModal(true);
    }
  }, [isLoading, uploadedFiles.length, searchParams]);

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
    const isNew = searchParams.get('new') === 'true';
    if (isNew || location.state?.newNotebook) {
      setShowUploadModal(true);
      // Clean up the state properly using React Router
      navigate(`/chat/${chatId}`, { replace: true, state: {} });
      return;
    }
    
    if (filesData?.files?.length > 0) {
      const currentFiles = filesData.files;
      const targetId = activeFileId || location.state?.activeFileId;
      
      if (!targetId) {
        // No active file, set to the first one
        const firstFile = currentFiles[0];
        setActiveFileId(firstFile.file_id);
        setActiveDocument(firstFile.filename);
        setSelectedFiles(prev => !prev.includes(firstFile.file_id) ? [...prev, firstFile.file_id] : prev);
      } else {
        const activeFile = currentFiles.find((f: any) => f.file_id === targetId);
        if (activeFile) {
          setActiveDocument(activeFile.filename);
          setSelectedFiles(prev => !prev.includes(activeFile.file_id) ? [...prev, activeFile.file_id] : prev);
        }
      }
    } else if (filesData?.files?.length === 0) {
      setActiveFileId(null);
      setActiveDocument(null);
      setSelectedFiles([]);
    }
  }, [filesData?.files, activeFileId, location.state, searchParams, navigate, chatId]);

  const handleFilesUploaded = async (files: File[]) => {
    try {
      let lastUploadedId = null;
      let newFileIds: string[] = [];
      for (const file of files) {
        const resp = await uploadMutation.mutateAsync({ file, chatId });
        if (resp && resp.file_id) {
            lastUploadedId = resp.file_id;
            newFileIds.push(resp.file_id);
        }
      }
      await refetch();
      setShowUploadModal(false);
      
      if (lastUploadedId) {
          setActiveFileId(lastUploadedId);
          setSelectedFiles(prev => [...new Set([...prev, ...newFileIds])]);
      }
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

  if (isLoading && uploadedFiles.length === 0) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-background">
         <div className="animate-pulse flex flex-col items-center">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center mb-4">
               <FileText className="w-4 h-4 text-white" />
            </div>
            <p className="text-gray-400">Loading chat...</p>
         </div>
      </div>
    );
  }

  return (
    <div className="font-sans bg-background flex flex-col h-screen">
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
          .font-sans { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
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

      {/* Navbar (Desktop + Mobile) */}
      <nav className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50 flex-none h-16 flex items-center justify-between px-4 md:px-6">
        <div className="flex items-center gap-4 w-full justify-between">
          <div 
            className="text-xl font-bold tracking-tight cursor-pointer flex items-center gap-2"
            onClick={() => navigate("/dashboard")}
          >
            <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <span className="hidden sm:inline-block">LegalMind</span>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400 hidden md:block">
              {user?.displayName || user?.email}
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="rounded-full h-9 w-9 bg-white/5 border border-white/10 hover:bg-white/10">
                  {user?.photoURL ? (
                    <img src={user.photoURL} alt="Profile" className="w-full h-full rounded-full" referrerPolicy="no-referrer" />
                  ) : (
                    <UserIcon className="h-4 w-4 text-gray-300" />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48 bg-[#111] border-white/10 text-white">
                <DropdownMenuItem className="focus:bg-white/10 cursor-pointer" onClick={() => navigate("/")}>
                  Home
                </DropdownMenuItem>
                <DropdownMenuItem className="focus:bg-white/10 cursor-pointer" onClick={() => navigate("/dashboard")}>
                  Dashboard
                </DropdownMenuItem>
                <DropdownMenuItem className="focus:bg-red-500/20 text-red-400 cursor-pointer" onClick={handleLogout}>
                  <LogOut className="h-4 w-4 mr-2" />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            <button onClick={() => setIsMobileMenuOpen((prev) => !prev)} className="md:hidden z-30 ml-2">
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
          </div>
        </div>
      </nav>

      {isMobileMenuOpen && (
        <div className="md:hidden fixed inset-0 top-16 z-20 bg-background p-6 pt-8">
          <nav className="flex flex-col items-start space-y-4">
            <a href="#" onClick={(e) => { e.preventDefault(); navigate("/"); }} className="text-lg text-muted-foreground hover:text-foreground transition-colors">Home</a>
            <a href="#" onClick={(e) => { e.preventDefault(); navigate("/dashboard"); }} className="text-lg text-muted-foreground hover:text-foreground transition-colors">Dashboard</a>
            <a href="#" onClick={(e) => { e.preventDefault(); handleLogout(); }} className="text-lg text-red-400 hover:text-red-300 transition-colors">Log out</a>
          </nav>
        </div>
      )}

      {/* Main Content Area */}
      <main className="flex-1 flex flex-row" style={{ overflow: "hidden" }}>
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
            selectedFileIds={selectedFiles} // added support for multi-doc
            chatId={chatId}
            onConversationIdChange={handleConversationIdChange}
            showSourcesDesktop={showSourcesDesktop}
            setShowSourcesDesktop={setShowSourcesDesktop}
            showInsightsDesktop={showInsightsDesktop}
            setShowInsightsDesktop={setShowInsightsDesktop}
            chatStatus={statusData}
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