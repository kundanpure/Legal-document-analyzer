import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Plus, FileText, File } from "lucide-react";

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
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const getFileIcon = (type: string) => {
  if (type.includes('pdf')) return FileText;
  return File;
};

export const SourcesPanel = ({ 
  uploadedFiles, 
  selectedFiles, 
  onFileSelect, 
  onFileClick, 
  onAddMore 
}: SourcesPanelProps) => {
  return (
    <div className="w-80 bg-card border-r border-border flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <h2 className="heading-sans text-xl font-medium">Sources</h2>
          <Badge variant="secondary" className="text-xs">
            {uploadedFiles.length} / 50 PDFs
          </Badge>
        </div>
        
        <Button 
          onClick={onAddMore}
          className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20"
          variant="outline"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add More Documents
        </Button>
      </div>

      {/* Files List */}
      <div className="flex-1 overflow-y-auto">
        {uploadedFiles.length === 0 ? (
          <div className="p-6 text-center text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-sm">No documents uploaded yet</p>
            <p className="text-xs mt-1">Click "Add More Documents" to get started</p>
          </div>
        ) : (
          <div className="p-4 space-y-3">
            {uploadedFiles.map((file) => {
              const FileIcon = getFileIcon(file.type);
              const isSelected = selectedFiles.includes(file.id);
              
              return (
                <Card 
                  key={file.id}
                  className={`p-4 cursor-pointer transition-smooth border ${
                    isSelected 
                      ? 'bg-primary/10 border-primary/30' 
                      : 'bg-card border-border hover:bg-muted/50'
                  }`}
                  onClick={() => onFileClick(file)}
                >
                  <div className="flex items-start gap-3">
                    <Checkbox 
                      checked={isSelected}
                      onCheckedChange={(checked) => onFileSelect(file.id, !!checked)}
                      onClick={(e) => e.stopPropagation()}
                      className="mt-1"
                    />
                    
                    <FileIcon className={`h-5 w-5 mt-0.5 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
                    
                    <div className="flex-1 min-w-0">
                      <h4 className={`text-sm font-medium truncate ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                        {file.name}
                      </h4>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-muted-foreground">
                          {formatFileSize(file.size)}
                        </span>
                        <span className="text-xs text-muted-foreground">•</span>
                        <span className="text-xs text-muted-foreground">
                          {file.uploadedAt.toLocaleDateString()}
                        </span>
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
      <div className="p-4 border-t border-border bg-muted/30">
        <p className="text-xs text-muted-foreground text-center">
          Supported: Contracts, Loan Agreements,<br />
          Rental Agreements, Terms & Conditions
        </p>
      </div>
    </div>
  );
};