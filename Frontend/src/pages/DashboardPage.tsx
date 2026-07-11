import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useFiles } from "@/hooks/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Plus, FileText, ArrowRight, Loader2, LogOut, User as UserIcon, ArrowLeft } from "lucide-react";
import { logout } from "@/lib/firebase";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const DashboardPage = () => {
  const navigate = useNavigate();
  const { user, loading } = useAuth();
  const { data: filesData, isLoading: filesLoading } = useFiles();

  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

  const handleLogout = async () => {
    try {
      await logout();
      navigate("/");
      toast.success("Logged out successfully");
    } catch (error) {
      toast.error("Error logging out");
    }
  };

  if (loading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#050505]">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  const notebooks = filesData?.files || [];

  return (
    <div className="min-h-screen bg-[#050505] text-white selection:bg-blue-500/30">
      {/* Navbar */}
      <nav className="border-b border-white/10 bg-black/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div 
            className="text-xl font-bold tracking-tight cursor-pointer flex items-center gap-2"
            onClick={() => navigate("/")}
          >
            <div className="w-8 h-8 rounded bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
              <FileText className="w-4 h-4 text-white" />
            </div>
            LegalMind
          </div>
          
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost" 
              onClick={() => navigate("/")}
              className="text-gray-400 hover:text-white hidden md:flex"
            >
              Home
            </Button>
            <span className="text-sm text-gray-400 hidden md:block">
              {user.displayName || user.email}
            </span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="rounded-full h-9 w-9 bg-white/5 border border-white/10 hover:bg-white/10">
                  {user.photoURL ? (
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
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-10">
          <Button
            onClick={() => navigate("/")}
            variant="ghost"
            size="sm"
            className="flex items-center gap-2 mb-6 px-3 py-1 border rounded-md w-fit"
            style={{
              background: "rgba(255,255,255,0.02)",
              borderColor: "rgba(255,255,255,0.12)",
              color: "#e5e7eb",
            }}
          >
            <ArrowLeft className="h-4 w-4" />
            Go Back
          </Button>
          <h1 className="text-3xl font-bold mb-2">My Notebooks</h1>
          <p className="text-gray-400">Manage your legal documents and chat sessions.</p>
        </div>

        {filesLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {/* Create New Notebook Card */}
            <Card 
              onClick={() => navigate("/chat", { state: { newNotebook: true } })}
              className="group cursor-pointer border border-white/10 bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/20 transition-all duration-300 rounded-2xl h-56 flex flex-col items-center justify-center shadow-lg hover:shadow-xl hover:-translate-y-1"
            >
              <div className="w-14 h-14 rounded-full bg-blue-500/10 flex items-center justify-center mb-4 group-hover:scale-110 group-hover:bg-blue-500/20 transition-all">
                <Plus className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="font-medium text-lg">New Notebook</h3>
              <p className="text-sm text-gray-500 mt-1">Upload a new document</p>
            </Card>

            {/* Existing Notebooks */}
            {notebooks.map((notebook: any) => (
              <Card 
                key={notebook.file_id}
                onClick={() => navigate("/chat", { state: { activeFileId: notebook.file_id } })}
                className="group cursor-pointer border border-white/10 bg-[#0A0A0A] hover:border-white/20 transition-all duration-300 rounded-2xl h-56 flex flex-col p-6 shadow-lg hover:shadow-xl hover:-translate-y-1 relative overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-bl-full -mr-10 -mt-10 transition-all group-hover:bg-blue-500/10" />
                
                <div className="flex items-start justify-between mb-auto relative z-10">
                  <div className="p-3 rounded-lg bg-white/5 border border-white/10">
                    <FileText className="w-5 h-5 text-blue-400" />
                  </div>
                </div>

                <div className="relative z-10">
                  <h3 className="font-medium text-base line-clamp-2 mb-1 group-hover:text-blue-400 transition-colors">
                    {notebook.filename}
                  </h3>
                  <div className="flex items-center text-xs text-gray-500 justify-between">
                    <span>{new Date(notebook.uploaded_at).toLocaleDateString()}</span>
                    <ArrowRight className="w-4 h-4 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all" />
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default DashboardPage;
