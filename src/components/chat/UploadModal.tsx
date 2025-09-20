import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFilesUploaded: (files: File[]) => void;
  currentFileCount: number;
  isUploading?: boolean;
}

const MAX_FILES = 50;
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_TYPES = ['application/pdf'];

export const UploadModal = ({ 
  isOpen, 
  onClose, 
  onFilesUploaded, 
  currentFileCount,
  isUploading = false
}: UploadModalProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  const validateFile = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return `${file.name}: Only PDF files are supported`;
    }
    if (file.size > MAX_FILE_SIZE) {
      return `${file.name}: File size must be under 10MB`;
    }
    if (currentFileCount + selectedFiles.length >= MAX_FILES) {
      return `Cannot exceed ${MAX_FILES} files total`;
    }
    return null;
  };

  const handleFiles = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const newErrors: string[] = [];
    const validFiles: File[] = [];

    // Filter out duplicates
    const existingNames = selectedFiles.map(f => f.name);
    
    fileArray.forEach(file => {
      if (existingNames.includes(file.name)) {
        newErrors.push(`${file.name}: File already selected`);
        return;
      }
      
      const error = validateFile(file);
      if (error) {
        newErrors.push(error);
      } else {
        validFiles.push(file);
      }
    });

    setErrors(prev => [...prev.filter(e => !fileArray.some(f => e.includes(f.name))), ...newErrors]);
    setSelectedFiles(prev => [...prev, ...validFiles]);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
      e.dataTransfer.clearData();
    }
  }, [selectedFiles, currentFileCount]);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
      e.target.value = '';
    }
  };

  const removeFile = (index: number) => {
    const removedFile = selectedFiles[index];
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
    // Remove related errors
    setErrors(prev => prev.filter(error => !error.includes(removedFile.name)));
  };

  const clearAllErrors = () => {
    setErrors([]);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    
    try {
      // Clear any existing errors
      clearAllErrors();
      
      // Call the parent function
      await onFilesUploaded(selectedFiles);
      
      // Reset state on successful upload
      setSelectedFiles([]);
      setErrors([]);
    } catch (error: any) {
      // Handle upload errors
      setErrors([`Upload failed: ${error?.message || 'Please try again'}`]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const canUpload = selectedFiles.length > 0 && !isUploading && currentFileCount + selectedFiles.length <= MAX_FILES;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="heading-sans text-2xl font-light">
            Upload Legal Documents
          </DialogTitle>
          <div className="flex items-center gap-4 mt-3">
            <Badge variant="secondary">
              {currentFileCount + selectedFiles.length} / {MAX_FILES} PDFs
            </Badge>
            <p className="text-sm text-muted-foreground font-light">
              Max 10MB per file
            </p>
          </div>
        </DialogHeader>

        <div className="space-y-6">
          {/* Supported Documents Info */}
          <Card className="p-6 bg-gradient-primary/3 border-0 shadow-soft">
            <h4 className="font-medium text-sm mb-3 text-foreground">Supported Documents</h4>
            <div className="flex flex-wrap gap-2">
              {["Contracts", "Loan Agreements", "Rental Agreements", "Terms & Conditions", "NDAs", "Employment Agreements"].map((type) => (
                <Badge key={type} variant="outline" className="text-xs font-light border-border/50">
                  {type}
                </Badge>
              ))}
            </div>
          </Card>

          {/* Upload Area */}
          <Card
            className={`border-2 border-dashed p-12 text-center cursor-pointer transition-smooth rounded-2xl ${
              dragActive 
                ? 'border-primary bg-gradient-primary/5 shadow-soft' 
                : 'border-border/50 hover:border-primary/30 hover:bg-gradient-primary/2 hover:shadow-soft'
            } ${isUploading ? 'opacity-50 cursor-not-allowed' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !isUploading && document.getElementById('file-input')?.click()}
          >
            {isUploading ? (
              <Loader2 className="h-16 w-16 mx-auto mb-6 text-primary animate-spin" />
            ) : (
              <Upload className={`h-16 w-16 mx-auto mb-6 ${
                dragActive ? 'text-primary' : 'text-muted-foreground'
              }`} />
            )}
            <h3 className="font-medium mb-3 text-lg">
              {isUploading ? 'Uploading files...' : dragActive ? 'Drop files here' : 'Upload PDF Documents'}
            </h3>
            <p className="text-muted-foreground mb-6 font-light leading-relaxed">
              {isUploading ? 'Please wait while your files are being processed' : 'Drag and drop your files here, or click to browse'}
            </p>
            {!isUploading && (
              <Button variant="outline" className="px-8 py-3 rounded-xl font-medium">
                Choose Files
              </Button>
            )}
            <input
              id="file-input"
              type="file"
              multiple
              accept=".pdf"
              onChange={handleFileInput}
              className="hidden"
              disabled={isUploading}
            />
          </Card>

          {/* Selected Files */}
          {selectedFiles.length > 0 && (
            <div className="max-h-40 overflow-y-auto">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-sm">Selected Files ({selectedFiles.length})</h4>
                {!isUploading && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedFiles([])}
                    className="text-xs text-muted-foreground hover:text-foreground"
                  >
                    Clear All
                  </Button>
                )}
              </div>
              <div className="space-y-3">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="flex items-center gap-4 p-4 bg-card rounded-xl border-0 shadow-soft">
                    <FileText className="h-5 w-5 text-primary" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{file.name}</p>
                      <p className="text-xs text-muted-foreground font-light">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                    {isUploading ? (
                      <Loader2 className="h-4 w-4 text-primary animate-spin" />
                    ) : (
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => removeFile(index)}
                        className="h-8 w-8 p-0 rounded-lg hover:bg-muted"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Errors */}
          {errors.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-sm text-destructive">Upload Errors ({errors.length})</h4>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={clearAllErrors}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear
                </Button>
              </div>
              <div className="max-h-32 overflow-y-auto space-y-2">
                {errors.map((error, index) => (
                  <div key={index} className="flex items-start gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
                    <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-destructive">{error}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-4 pt-6 border-t border-border/50">
            <Button 
              variant="outline" 
              onClick={onClose} 
              disabled={isUploading} 
              className="px-8 py-3 rounded-xl font-medium"
            >
              {isUploading ? 'Please Wait...' : 'Cancel'}
            </Button>
            <Button 
              onClick={handleUpload} 
              disabled={!canUpload}
              className="bg-gradient-primary hover:shadow-glow-primary px-8 py-3 rounded-xl font-medium"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  Uploading...
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Upload {selectedFiles.length} File{selectedFiles.length !== 1 ? 's' : ''}
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};