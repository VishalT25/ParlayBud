import { useEffect, useRef } from "react";
import { Terminal as TerminalIcon, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

interface TerminalProps {
  logs: string[];
  isProcessing?: boolean;
}

export const Terminal = ({ logs, isProcessing = false }: TerminalProps) => {
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-full rounded-2xl overflow-hidden bg-gradient-to-br from-card to-secondary/30 border border-border/50 shadow-lg">
      {/* Terminal Header */}
      <div className="flex items-center justify-between px-5 py-3 bg-gradient-to-r from-foreground to-foreground/90 border-b border-border/20">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full bg-destructive/80" />
            <div className="w-3 h-3 rounded-full bg-warning/80" />
            <div className="w-3 h-3 rounded-full bg-success/80" />
          </div>
          <div className="flex items-center gap-2 ml-2">
            <TerminalIcon className="w-4 h-4 text-background" />
            <span className="text-sm font-bold text-background tracking-wide">
              Processing Output
            </span>
          </div>
        </div>
        
        {isProcessing && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-background/20 backdrop-blur-sm">
            <Activity className="w-3.5 h-3.5 text-success animate-pulse-glow" />
            <span className="text-xs font-semibold text-background">Running...</span>
          </div>
        )}
      </div>

      {/* Terminal Content */}
      <div
        ref={terminalRef}
        className={cn(
          "font-mono text-sm p-5 h-72 overflow-y-auto",
          "bg-gradient-to-b from-background/95 to-background",
          "scrollbar-thin scrollbar-thumb-primary/30 scrollbar-track-transparent hover:scrollbar-thumb-primary/50"
        )}
      >
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
            <div className="p-4 rounded-xl bg-muted/30">
              <TerminalIcon className="w-8 h-8 text-muted-foreground/50" />
            </div>
            <div>
              <p className="text-muted-foreground font-semibold mb-1">
                Awaiting file upload...
              </p>
              <p className="text-xs text-muted-foreground/70">
                Processing logs will appear here
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            {logs.map((log, index) => {
              const isError = log.includes("[ERROR]");
              const isSuccess = log.includes("[SUCCESS]");
              const isInfo = log.includes("[INFO]");
              
              return (
                <div
                  key={index}
                  className={cn(
                    "py-1 px-2 rounded transition-colors duration-200 animate-fade-in",
                    isError && "text-destructive bg-destructive/5",
                    isSuccess && "text-success bg-success/5 font-semibold",
                    isInfo && "text-primary bg-primary/5",
                    !isError && !isSuccess && !isInfo && "text-foreground/80"
                  )}
                  style={{ 
                    animationDelay: `${index * 30}ms`,
                    animationFillMode: 'backwards'
                  }}
                >
                  <span className="select-text">{log}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
