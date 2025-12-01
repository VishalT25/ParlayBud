import { useCallback, useState } from "react";
import { Upload, FileSpreadsheet, X, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
}

export const FileUploader = ({ onFileSelect }: FileUploaderProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files);
      const validFile = files.find(
        (file) =>
          file.name.endsWith(".xlsx") || 
          file.name.endsWith(".xls") ||
          file.name.endsWith(".json")
      );

      if (validFile) {
        setSelectedFile(validFile);
        onFileSelect(validFile);
      }
    },
    [onFileSelect]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files[0]) {
        setSelectedFile(files[0]);
        onFileSelect(files[0]);
      }
    },
    [onFileSelect]
  );

  const clearFile = useCallback(() => {
    setSelectedFile(null);
  }, []);

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 group",
          isDragging
            ? "border-primary bg-primary/10 scale-[1.02] shadow-lg shadow-primary/20"
            : "border-border/50 bg-gradient-to-br from-card to-muted/20 hover:border-primary/50 hover:shadow-md"
        )}
      >
        <input
          type="file"
          accept=".xlsx,.xls,.json"
          onChange={handleFileInput}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          id="file-upload"
        />

        <div className="flex flex-col items-center justify-center gap-5 pointer-events-none">
          {selectedFile ? (
            <>
              <div className="relative">
                <div className="p-5 rounded-2xl bg-gradient-to-br from-primary to-accent shadow-lg">
                  <FileSpreadsheet className="w-10 h-10 text-white" />
                </div>
                <div className="absolute -top-1 -right-1">
                  <Sparkles className="w-5 h-5 text-success animate-pulse-glow" />
                </div>
              </div>
              
              <div className="text-center space-y-2">
                <p className="text-lg font-bold text-foreground">
                  {selectedFile.name}
                </p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024).toFixed(2)} KB • Ready to process
                </p>
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={clearFile}
                className="pointer-events-auto border-destructive/50 text-destructive hover:bg-destructive/10 hover:border-destructive"
              >
                <X className="w-4 h-4 mr-2" />
                Remove File
              </Button>
            </>
          ) : (
            <>
              <div className="relative">
                <div className={cn(
                  "p-5 rounded-2xl bg-gradient-to-br from-primary/20 to-accent/10 transition-all duration-300",
                  isDragging ? "scale-110 from-primary/30 to-accent/20" : "group-hover:scale-105"
                )}>
                  <Upload className={cn(
                    "w-10 h-10 transition-colors duration-300",
                    isDragging ? "text-primary" : "text-primary/70 group-hover:text-primary"
                  )} />
                </div>
                {isDragging && (
                  <div className="absolute inset-0 rounded-2xl bg-primary/20 animate-pulse-glow" />
                )}
              </div>
              
              <div className="text-center space-y-2">
                <p className="text-xl font-bold text-foreground">
                  {isDragging ? "Drop it like it's hot! 🔥" : "Drop your file here"}
                </p>
                <p className="text-sm text-muted-foreground">
                  or <span className="text-primary font-semibold">click to browse</span>
                </p>
              </div>
              
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-primary/5 border border-primary/20">
                  <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                  <span className="font-medium">.xlsx</span>
                </div>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-accent/5 border border-accent/20">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  <span className="font-medium">.json</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
