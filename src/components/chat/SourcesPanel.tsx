import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Plus, FileText, File, ArrowLeft } from "lucide-react";

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

interface SourcesPanelProps {
  uploadedFiles: UploadedFile[];
  selectedFiles: string[];
  onFileSelect: (fileId: string, selected: boolean) => void;
  onFileClick: (file: UploadedFile) => void;
  onAddMore: () => void;
}

const formatFileSize = (bytes: number) => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};

const getFileIcon = (type: string) => {
  if (type.includes("pdf")) return FileText;
  return File;
};

export const SourcesPanel = ({
  uploadedFiles,
  selectedFiles,
  onFileSelect,
  onFileClick,
  onAddMore,
}: SourcesPanelProps) => {
  return (
    <div
      className="w-80 flex flex-col h-full text-gray-100 border-r"
      style={{
        background: "linear-gradient(180deg,#000,#050505)",
        borderColor: "rgba(255,255,255,0.08)",
      }}
    >
      <style>{`
        .src-card {
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 12px;
          background: rgba(255,255,255,0.02);
          transition: all 0.25s ease;
        }
        .src-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 6px 16px rgba(0,0,0,0.5), 0 0 8px rgba(255,255,255,0.08);
          border-color: rgba(255,255,255,0.18);
        }
        .dark-scrollbar::-webkit-scrollbar { width: 8px; }
        .dark-scrollbar::-webkit-scrollbar-track { background: #050505; }
        .dark-scrollbar::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg,#2b2b2b,#141414);
          border-radius: 999px;
          border: 2px solid rgba(0,0,0,0.6);
        }
      `}</style>

      {/* Header */}
      <div className="p-6 border-b border-[rgba(255,255,255,0.08)] src-card m-2">
        {/* Go Back Button */}
        <Button
          onClick={() => (window.location.href = "/")}
          variant="ghost"
          size="sm"
          className="flex items-center gap-2 mb-4 px-3 py-1 border rounded-md"
          style={{
            background: "rgba(255,255,255,0.02)",
            borderColor: "rgba(255,255,255,0.12)",
            color: "#e5e7eb",
          }}
        >
          <ArrowLeft className="h-4 w-4" />
          Go Back
        </Button>

        <div className="flex items-center justify-between mb-4">
          <h2 className="heading-sans text-xl font-semibold">Sources</h2>
          <Badge
            variant="secondary"
            className="text-xs border"
            style={{
              background: "rgba(255,255,255,0.05)",
              borderColor: "rgba(255,255,255,0.12)",
              color: "#9ca3af",
            }}
          >
            {uploadedFiles.length} / 50 PDFs
          </Badge>
        </div>

        <Button
          onClick={onAddMore}
          className="w-full border flex items-center justify-center gap-2"
          variant="outline"
          style={{
            background: "rgba(255,255,255,0.02)",
            borderColor: "rgba(255,255,255,0.12)",
            color: "#e5e7eb",
          }}
        >
          <Plus className="h-4 w-4" />
          Add More Documents
        </Button>
      </div>

      {/* Files List */}
      <div className="flex-1 overflow-y-auto dark-scrollbar px-2 pb-2">
        {uploadedFiles.length === 0 ? (
          <div className="p-6 text-center text-gray-400 src-card">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-40" />
            <p className="text-sm">No documents uploaded yet</p>
            <p className="text-xs mt-1">Click "Add More Documents" to get started</p>
          </div>
        ) : (
          <div className="space-y-3">
            {uploadedFiles.map((file) => {
              const FileIcon = getFileIcon(file.type);
              const isSelected = selectedFiles.includes(file.id);

              return (
                <Card
                  key={file.id}
                  className="p-4 cursor-pointer src-card"
                  style={{
                    borderColor: isSelected
                      ? "rgba(255,255,255,0.25)"
                      : "rgba(255,255,255,0.12)",
                  }}
                  onClick={() => onFileClick(file)}
                >
                  <div className="flex items-start gap-3">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={(checked) => onFileSelect(file.id, !!checked)}
                      onClick={(e) => e.stopPropagation()}
                      className="mt-1"
                    />

                    <FileIcon
                      className={`h-5 w-5 mt-0.5 ${
                        isSelected ? "text-white" : "text-gray-500"
                      }`}
                    />

                    <div className="flex-1 min-w-0">
                      <h4
                        className={`text-sm font-medium truncate ${
                          isSelected ? "text-white" : "text-gray-200"
                        }`}
                      >
                        {file.name}
                      </h4>
                      <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                        <span>{formatFileSize(file.size)}</span>
                        <span>•</span>
                        <span>{file.uploadedAt.toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-[rgba(255,255,255,0.08)] src-card m-2">
        <p className="text-xs text-gray-500 text-center">
          Supported: Contracts, Loan Agreements,
          <br />
          Rental Agreements, Terms & Conditions
        </p>
      </div>
    </div>
  );
};
