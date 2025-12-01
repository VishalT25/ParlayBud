import { LayoutGrid, List, LayoutList } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export type ViewMode = "detailed" | "compact" | "list";

interface ViewModeToggleProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
}

export function ViewModeToggle({ viewMode, onViewModeChange }: ViewModeToggleProps) {
  const modes: { value: ViewMode; icon: any; label: string }[] = [
    { value: "detailed", icon: LayoutList, label: "Detailed" },
    { value: "compact", icon: LayoutGrid, label: "Compact" },
    { value: "list", icon: List, label: "List" },
  ];

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-muted/50 border border-border/50">
      {modes.map((mode) => {
        const Icon = mode.icon;
        const isActive = viewMode === mode.value;
        
        return (
          <Button
            key={mode.value}
            variant="ghost"
            size="sm"
            onClick={() => onViewModeChange(mode.value)}
            className={cn(
              "gap-2 transition-all duration-300",
              isActive
                ? "bg-gradient-to-r from-primary to-accent text-white shadow-md"
                : "hover:bg-background/50"
            )}
          >
            <Icon className="w-4 h-4" />
            <span className="hidden sm:inline text-xs font-semibold">{mode.label}</span>
          </Button>
        );
      })}
    </div>
  );
}
