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
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["application/pdf"];

export const UploadModal = ({
  isOpen,
  onClose,
  onFilesUploaded,
  currentFileCount,
  isUploading = false,
}: UploadModalProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  const validateFile = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) return `${file.name}: Only PDF files are supported`;
    if (file.size > MAX_FILE_SIZE) return `${file.name}: File size must be under 10MB`;
    if (currentFileCount + selectedFiles.length >= MAX_FILES) return `Cannot exceed ${MAX_FILES} files total`;
    return null;
  };

  const handleFiles = (files: FileList | File[]) => {
    const fileArray = Array.from(files);
    const newErrors: string[] = [];
    const validFiles: File[] = [];
    const existingNames = selectedFiles.map((f) => f.name);

    fileArray.forEach((file) => {
      if (existingNames.includes(file.name)) {
        newErrors.push(`${file.name}: File already selected`);
        return;
      }
      const error = validateFile(file);
      if (error) newErrors.push(error);
      else validFiles.push(file);
    });

    setErrors((prev) => [...prev.filter((e) => !fileArray.some((f) => e.includes(f.name))), ...newErrors]);
    setSelectedFiles((prev) => [...prev, ...validFiles]);
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
        e.dataTransfer.clearData();
      }
    },
    [selectedFiles, currentFileCount] // Added dependencies
  );

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
      e.target.value = "";
    }
  };

  const removeFile = (index: number) => {
    const removedFile = selectedFiles[index];
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setErrors((prev) => prev.filter((error) => !error.includes(removedFile.name)));
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    try {
      setErrors([]);
      await onFilesUploaded(selectedFiles);
      setSelectedFiles([]);
    } catch (error: any) {
      setErrors([`Upload failed: ${error?.message || "Please try again"}`]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  };

  const canUpload = selectedFiles.length > 0 && !isUploading && currentFileCount + selectedFiles.length <= MAX_FILES;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        // Simplified styles: h-[90vh] and my-auto center the 90vh modal in the 100vh viewport.
        className="w-[95vw] max-w-2xl md:max-w-3xl h-[90vh] flex flex-col mx-auto my-auto p-0 gap-0"
      >
        {/* Header (Sticky) */}
        <div className="flex-shrink-0 p-6 pb-4 border-b border-border/50">
          <DialogHeader>
            <DialogTitle className="heading-sans text-xl md:text-2xl font-light">
              Upload Legal Documents
            </DialogTitle>
            <div className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 mt-3">
              <Badge variant="secondary">
                {currentFileCount + selectedFiles.length} / {MAX_FILES} PDFs
              </Badge>
              <p className="text-sm text-muted-foreground font-light">Max 10MB per file</p>
            </div>
          </DialogHeader>
        </div>

        {/* Scrollable Body (Single Scrollbar) */}
        {/* MODIFIED: Removed max-w-2xl mx-auto w-full to let it fill the modal width */}
        <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4 md:py-6 space-y-6">
          {/* Upload Area */}
          <Card
            className={`border-2 border-dashed p-6 md:p-12 text-center cursor-pointer rounded-2xl transition ${
              dragActive ? "border-primary bg-primary/5" : "border-border/50 hover:border-primary/40"
            } ${isUploading ? "opacity-50 cursor-not-allowed" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !isUploading && document.getElementById("file-input")?.click()}
          >
            {isUploading ? (
              <Loader2 className="h-12 w-12 md:h-16 md:w-16 mx-auto mb-4 text-primary animate-spin" />
            ) : (
              <Upload className={`h-12 w-12 md:h-16 md:w-16 mx-auto mb-4 ${dragActive ? "text-primary" : "text-muted-foreground"}`} />
            )}
            <h3 className="font-medium text-base md:text-lg mb-2">
              {isUploading ? "Uploading files..." : dragActive ? "Drop files here" : "Upload PDF Documents"}
            </h3>
            <p className="text-sm md:text-base text-muted-foreground mb-4">
              {isUploading ? "Please wait while your files are being processed" : "Drag and drop your files here, or click to browse"}
            </p>
            {!isUploading && <Button variant="outline">Choose Files</Button>}
            <input id="file-input" type="file" multiple accept=".pdf" onChange={handleFileInput} className="hidden" disabled={isUploading} />
          </Card>

          {/* Selected Files */}
          {selectedFiles.length > 0 && (
            <div>
              <h4 className="font-medium text-sm mb-3">Selected Files ({selectedFiles.length})</h4>
              {/* MODIFIED: Removed max-h-48 and overflow-y-auto */}
              <div className="space-y-2 pr-2">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="flex items-center gap-3 p-3 bg-card rounded-lg">
                    <FileText className="h-5 w-5 text-primary" />
                    <div className="flex-1 truncate">
                      <p className="text-sm">{file.name}</p>
                      <p className="text-xs text-muted-foreground">{formatFileSize(file.size)}</p>
                    </div>
                    {!isUploading && (
                      <Button size="sm" variant="ghost" onClick={() => removeFile(index)}>
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
            <div>
              <h4 className="font-medium text-sm text-destructive mb-2">Upload Errors</h4>
              {/* MODIFIED: Removed max-h-32 and overflow-y-auto */}
              <div className="space-y-2 pr-2">
                {errors.map((err, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 bg-destructive/10 rounded-lg text-xs text-destructive">
                    <AlertCircle className="h-4 w-4" /> {err}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer (Sticky) */}
        <div className="flex-shrink-0 p-6 border-t border-border/50">
          <div className="flex flex-col md:flex-row justify-end gap-3">
            <Button variant="outline" onClick={onClose} disabled={isUploading}>
              {isUploading ? "Please Wait..." : "Cancel"}
            </Button>
            <Button onClick={handleUpload} disabled={!canUpload}>
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin mr-2" /> Uploading...
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 mr-2" /> Upload {selectedFiles.length} File{selectedFiles.length !== 1 ? "s" : ""}
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};