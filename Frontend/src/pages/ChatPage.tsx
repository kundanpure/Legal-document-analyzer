import { useState, useEffect } from "react";
import { useNavigate, useLocation, useParams, useSearchParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { ChatSection } from "@/components/chat/ChatSection";
import { InsightsPanel } from "@/components/chat/InsightsPanel";
import { UploadModal } from "@/components/chat/UploadModal";
import { useFiles, useUploadFile } from "@/hooks/api";
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

const StartupGuide = ({ backendUrl }: { backendUrl: string }) => {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'starting' | 'ready' | 'error'>('checking');
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedTime(prev => prev + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch(`${backendUrl}/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });
        
        if (response.ok) {
          setBackendStatus('ready');
        } else {
          setBackendStatus('starting');
        }
      } catch (error) {
        setBackendStatus('starting');
      }
    };

    // Initial check
    checkBackendStatus();

    // Check every 5 seconds
    const interval = setInterval(checkBackendStatus, 5000);

    return () => clearInterval(interval);
  }, [backendUrl]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStatusIcon = () => {
    switch (backendStatus) {
      case 'checking':
        return <Clock className="w-6 h-6 text-blue-500 animate-pulse" />;
      case 'starting':
        return <Server className="w-6 h-6 text-orange-500 animate-spin" />;
      case 'ready':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-6 h-6 text-red-500" />;
      default:
        return <Clock className="w-6 h-6 text-blue-500" />;
    }
  };

  const getStatusMessage = () => {
    switch (backendStatus) {
      case 'checking':
        return 'Checking backend server status...';
      case 'starting':
        return 'Backend server is starting up...';
      case 'ready':
        return 'Backend server is ready!';
      case 'error':
        return 'Having trouble connecting to the backend server';
      default:
        return 'Initializing...';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black flex items-center justify-center p-4 font-sans">
      <div className="max-w-2xl w-full bg-gray-800/40 backdrop-blur-lg rounded-2xl border border-gray-700/50 shadow-2xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Zap className="w-8 h-8 text-blue-400 mr-2" />
            <h1 className="text-3xl font-bold text-white">Legal AI Assistant</h1>
          </div>
          <p className="text-gray-300 text-lg">Getting everything ready for you...</p>
        </div>

        {/* Status Section */}
        <div className="bg-gray-800/50 rounded-xl p-6 mb-6 border border-gray-700/30">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              {getStatusIcon()}
              <span className="text-white font-medium">{getStatusMessage()}</span>
            </div>
            <div className="text-gray-400 font-mono text-sm">
              {formatTime(elapsedTime)}
            </div>
          </div>
          
          {/* Progress Bar */}
          <div className="w-full bg-gray-700/50 rounded-full h-2 mb-4">
            <div 
              className={`h-2 rounded-full transition-all duration-500 ${
                backendStatus === 'ready' 
                  ? 'bg-green-500 w-full' 
                  : 'bg-blue-500 w-1/2 animate-pulse'
              }`}
            />
          </div>

          {backendStatus === 'ready' && (
            <div className="text-center">
              <p className="text-green-400 font-medium">🎉 All systems ready! Redirecting...</p>
            </div>
          )}
        </div>

        {/* Information Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-700/40">
            <h3 className="text-white font-semibold mb-2 flex items-center">
              <Server className="w-4 h-4 mr-2 text-blue-400" />
              Backend Hosting
            </h3>
            <p className="text-gray-300 text-sm">
              We're using free hosting for our backend services. This means the server goes to sleep when not in use to save resources.
            </p>
          </div>

          <div className="bg-gray-800/40 rounded-lg p-4 border border-gray-700/40">
            <h3 className="text-white font-semibold mb-2 flex items-center">
              <Clock className="w-4 h-4 mr-2 text-blue-400" />
              Startup Time
            </h3>
            <p className="text-gray-300 text-sm">
              The first request may take 30-60 seconds as the server wakes up and initializes all services.
            </p>
          </div>
        </div>

        {/* What's Happening Section */}
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700/30">
          <h3 className="text-white font-semibold mb-4 text-center">What's happening behind the scenes?</h3>
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${backendStatus !== 'checking' ? 'bg-green-500' : 'bg-blue-500 animate-pulse'}`} />
              <span className="text-gray-300 text-sm">Waking up the backend server</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${backendStatus === 'ready' ? 'bg-green-500' : backendStatus === 'starting' ? 'bg-blue-500 animate-pulse' : 'bg-gray-600'}`} />
              <span className="text-gray-300 text-sm">Loading AI models and dependencies</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${backendStatus === 'ready' ? 'bg-green-500' : 'bg-gray-600'}`} />
              <span className="text-gray-300 text-sm">Preparing your document analysis environment</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-6">
          <p className="text-gray-400 text-sm">
            Thank you for your patience! This one-time setup ensures optimal performance.
          </p>
          {backendStatus === 'starting' && (
            <p className="text-blue-400 text-xs mt-2 animate-pulse">
              Average wait time: 45 seconds • Server URL: {backendUrl}
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

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

  const backendUrl = "http://localhost:8000";
  const { toast } = useToast();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { fileId } = useParams<{ fileId: string }>();
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

  const { data: filesData, isLoading, error, refetch } = useFiles();
  const uploadMutation = useUploadFile();

  // Pick up activeFileId from URL params or navigation state
  useEffect(() => {
    if (fileId) {
      setActiveFileId(fileId);
      setSelectedFiles([fileId]);
    } else if (location.state?.activeFileId) {
      setActiveFileId(location.state.activeFileId);
      setSelectedFiles([location.state.activeFileId]);
    }
  }, [fileId, location.state]);

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
    const isNew = searchParams.get('new') === 'true';
    if (isNew || location.state?.newNotebook) {
      setShowUploadModal(true);
      // Clean up the state properly using React Router
      navigate('/chat', { replace: true, state: {} });
      return;
    }
    if (!fileId && !location.state?.activeFileId) {
      setActiveFileId(null);
      setActiveDocument(null);
      setSelectedFiles([]);
    } else if ((fileId || location.state?.activeFileId) && uploadedFiles.length > 0) {
      const targetId = fileId || location.state?.activeFileId;
      const activeFile = uploadedFiles.find(f => f.id === targetId);
      if (activeFile) {
        setActiveDocument(activeFile.name);
      }
    }
  }, [uploadedFiles, selectedFiles.length, fileId, location.state, searchParams, navigate]);

  const handleFilesUploaded = async (files: File[]) => {
    try {
      let lastUploadedId = null;
      for (const file of files) {
        const resp = await uploadMutation.mutateAsync(file);
        // Assuming your mutation returns the uploaded file_id
        if (resp && resp.file_id) {
            lastUploadedId = resp.file_id;
        }
      }
      await refetch();
      setShowUploadModal(false);
      
      // Navigate to the newly uploaded file if we got its ID
      if (lastUploadedId) {
          navigate(`/chat/${lastUploadedId}`);
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

  if (isLoading) {
    return <StartupGuide backendUrl={backendUrl} />;
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