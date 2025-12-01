import { useState } from "react";
import { ChevronDown, ChevronUp, TrendingUp, Trophy, Copy, Check, Zap, Target } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useToast } from "@/hooks/use-toast";

type ViewMode = "detailed" | "compact" | "list";

interface Leg {
  row_index: number;
  team: string;
  player: string;
  game: string;
  proposition: string;
  line: number;
  odds: number;
  implied_probability: number;
  model_probability: number;
}

interface Parlay {
  label: string;
  num_legs: number;
  ev: number;
  hit_probability: number;
  implied_decimal: number;
  implied_american: string;
  kelly_fraction: number;
  legs: Leg[];
}

interface ParlayCardProps {
  parlay: Parlay;
  index: number;
  viewMode?: ViewMode;
  isSelectable?: boolean;
  isSelected?: boolean;
  onToggleSelect?: () => void;
}

export const ParlayCard = ({ 
  parlay, 
  index, 
  viewMode = "detailed",
  isSelectable = false, 
  isSelected = false, 
  onToggleSelect 
}: ParlayCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const formatParlayAsText = () => {
    let text = `🎯 Parlay #${index + 1}\n`;
    text += `EV: ${(parlay.ev * 100).toFixed(2)}% | `;
    text += `Hit Prob: ${(parlay.hit_probability * 100).toFixed(2)}% | `;
    text += `Odds: ${parlay.implied_american}\n\n`;
    
    parlay.legs.forEach((leg, i) => {
      text += `Leg ${i + 1}: ${leg.player} (${leg.team})\n`;
      text += `  ${leg.proposition} @ ${leg.odds}\n`;
      text += `  Model: ${(leg.model_probability * 100).toFixed(1)}% | Market: ${(leg.implied_probability * 100).toFixed(1)}%\n\n`;
    });
    
    return text;
  };

  const handleCopy = async () => {
    const text = formatParlayAsText();
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast({
      title: "Copied!",
      description: "Parlay copied to clipboard",
    });
    setTimeout(() => setCopied(false), 2000);
  };

  const getEVGradient = (ev: number) => {
    if (ev >= 0.15) return "from-success/20 to-success-glow/10";
    if (ev >= 0.10) return "from-primary/20 to-accent/10";
    return "from-muted to-muted/50";
  };

  // List view - compact single row
  if (viewMode === "list") {
    return (
      <Card
        className={cn(
          "group relative overflow-hidden border-border/50 backdrop-blur-sm transition-all duration-300 hover:shadow-md",
          isSelected && "ring-2 ring-primary shadow-primary"
        )}
      >
        <div className="relative p-4">
          <div className="flex items-center gap-4">
            {isSelectable && (
              <input
                type="checkbox"
                checked={isSelected}
                onChange={onToggleSelect}
                className="w-5 h-5 rounded-md border-2 border-border text-primary focus:ring-2 focus:ring-primary cursor-pointer"
              />
            )}
            
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <Badge 
                variant="outline" 
                className="font-semibold text-xs shrink-0 border-primary text-primary"
              >
                #{index + 1}
              </Badge>
              
              <div className="flex items-center gap-4 flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-xl font-bold text-gradient">{parlay.implied_american}</p>
                  <Badge variant="secondary" className="text-xs">{parlay.num_legs}L</Badge>
                </div>
                
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1.5">
                    <TrendingUp className="w-4 h-4 text-success" />
                    <span className="font-bold text-success">+{(parlay.ev * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Target className="w-4 h-4 text-primary" />
                    <span className="font-semibold">{(parlay.hit_probability * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="shrink-0"
            >
              {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="shrink-0"
            >
              {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
            </Button>
          </div>
          
          {/* Expanded Legs */}
          {isExpanded && (
            <div className="mt-4 space-y-3 animate-fade-in">
              {parlay.legs.map((leg, idx) => {
                const edgePercent = ((leg.model_probability - leg.implied_probability) * 100).toFixed(1);
                const hasEdge = leg.model_probability > leg.implied_probability;
                
                return (
                  <div
                    key={idx}
                    className="group/leg p-5 rounded-xl bg-gradient-to-br from-secondary/50 to-muted/30 border border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-md"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <p className="font-bold text-lg text-foreground mb-1">{leg.player}</p>
                        <p className="text-sm text-muted-foreground font-medium">{leg.team}</p>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={cn(
                          "font-bold text-sm px-3 py-1",
                          leg.odds > 0 ? "border-warning text-warning" : "border-primary text-primary"
                        )}
                      >
                        {leg.odds > 0 ? "+" : ""}{leg.odds}
                      </Badge>
                    </div>

                    <div className="space-y-3">
                      <div className="p-3 rounded-lg bg-background/50 backdrop-blur-sm">
                        <p className="text-sm font-semibold text-foreground mb-1">
                          {leg.proposition}
                        </p>
                        <p className="text-xs text-muted-foreground">{leg.game}</p>
                      </div>

                      <div className="grid grid-cols-3 gap-3">
                        <div className="text-center p-2 rounded-lg bg-background/30">
                          <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                            Market
                          </p>
                          <p className="text-sm font-bold text-foreground">
                            {(leg.implied_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center p-2 rounded-lg bg-primary/10">
                          <p className="text-xs text-primary mb-1 uppercase tracking-wide font-medium">
                            Model
                          </p>
                          <p className="text-sm font-bold text-primary">
                            {(leg.model_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center p-2 rounded-lg bg-background/30">
                          <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                            Edge
                          </p>
                          <p className={cn(
                            "text-sm font-bold",
                            hasEdge ? "text-success" : "text-muted-foreground"
                          )}>
                            {hasEdge ? "+" : ""}{edgePercent}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>
    );
  }

  // Compact view - smaller cards
  if (viewMode === "compact") {
    return (
      <Card
        className={cn(
          "group relative overflow-hidden card-glow border-border/50 backdrop-blur-sm",
          "transition-all duration-500 ease-out",
          isSelected && "ring-2 ring-primary shadow-primary"
        )}
        style={{ 
          animationDelay: `${index * 80}ms`,
          animationFillMode: 'backwards'
        }}
      >
        <div className="relative p-5">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-start gap-3 flex-1">
              {isSelectable && (
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={onToggleSelect}
                  className="mt-1 w-5 h-5 rounded-md border-2 border-border text-primary focus:ring-2 focus:ring-primary cursor-pointer"
                />
              )}
              
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge 
                    variant="outline" 
                    className="font-semibold text-xs border-primary text-primary"
                  >
                    {parlay.label.replace(/_/g, " ").toUpperCase()}
                  </Badge>
                </div>
                
                <div className="flex items-center gap-3">
                  <p className="text-2xl font-bold text-gradient">{parlay.implied_american}</p>
                  <Badge variant="secondary" className="text-xs">{parlay.num_legs} Legs</Badge>
                </div>
              </div>
            </div>

            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {copied ? <Check className="w-4 h-4 text-success" /> : <Copy className="w-4 h-4" />}
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-3 p-3 rounded-lg bg-gradient-to-br from-primary/10 to-accent/5 mb-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-success" />
              <div>
                <p className="text-xs text-muted-foreground">EV</p>
                <p className="text-lg font-bold text-success">+{(parlay.ev * 100).toFixed(1)}%</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Target className="w-4 h-4 text-primary" />
              <div>
                <p className="text-xs text-muted-foreground">Hit</p>
                <p className="text-lg font-bold">{(parlay.hit_probability * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
          
          {/* Expand Button */}
          <Button
            variant="ghost"
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full justify-between hover:bg-secondary/80 transition-all duration-300 group/btn"
          >
            <span className="text-sm font-semibold flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              {isExpanded ? "Hide" : "Show"} Leg Details
            </span>
            <div className="transition-transform duration-300 group-hover/btn:translate-x-1">
              {isExpanded ? (
                <ChevronUp className="w-5 h-5" />
              ) : (
                <ChevronDown className="w-5 h-5" />
              )}
            </div>
          </Button>

          {/* Expanded Legs */}
          {isExpanded && (
            <div className="mt-4 space-y-3 animate-fade-in">
              {parlay.legs.map((leg, idx) => {
                const edgePercent = ((leg.model_probability - leg.implied_probability) * 100).toFixed(1);
                const hasEdge = leg.model_probability > leg.implied_probability;
                
                return (
                  <div
                    key={idx}
                    className="group/leg p-5 rounded-xl bg-gradient-to-br from-secondary/50 to-muted/30 border border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-md"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <p className="font-bold text-lg text-foreground mb-1">{leg.player}</p>
                        <p className="text-sm text-muted-foreground font-medium">{leg.team}</p>
                      </div>
                      <Badge 
                        variant="outline" 
                        className={cn(
                          "font-bold text-sm px-3 py-1",
                          leg.odds > 0 ? "border-warning text-warning" : "border-primary text-primary"
                        )}
                      >
                        {leg.odds > 0 ? "+" : ""}{leg.odds}
                      </Badge>
                    </div>

                    <div className="space-y-3">
                      <div className="p-3 rounded-lg bg-background/50 backdrop-blur-sm">
                        <p className="text-sm font-semibold text-foreground mb-1">
                          {leg.proposition}
                        </p>
                        <p className="text-xs text-muted-foreground">{leg.game}</p>
                      </div>

                      <div className="grid grid-cols-3 gap-3">
                        <div className="text-center p-2 rounded-lg bg-background/30">
                          <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                            Market
                          </p>
                          <p className="text-sm font-bold text-foreground">
                            {(leg.implied_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center p-2 rounded-lg bg-primary/10">
                          <p className="text-xs text-primary mb-1 uppercase tracking-wide font-medium">
                            Model
                          </p>
                          <p className="text-sm font-bold text-primary">
                            {(leg.model_probability * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center p-2 rounded-lg bg-background/30">
                          <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                            Edge
                          </p>
                          <p className={cn(
                            "text-sm font-bold",
                            hasEdge ? "text-success" : "text-muted-foreground"
                          )}>
                            {hasEdge ? "+" : ""}{edgePercent}%
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>
    );
  }

  // Detailed view - original full layout
  return (
    <Card
      className={cn(
        "group relative overflow-hidden card-glow border-border/50 backdrop-blur-sm",
        "transition-all duration-500 ease-out",
        isSelected && "ring-2 ring-primary shadow-primary"
      )}
      style={{ 
        animationDelay: `${index * 80}ms`,
        animationFillMode: 'backwards'
      }}
    >
      <div className="relative p-6">
        {/* Header Section */}
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-start gap-4 flex-1">
            {isSelectable && (
              <div className="mt-1">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={onToggleSelect}
                  className="w-5 h-5 rounded-md border-2 border-border text-primary focus:ring-2 focus:ring-primary focus:ring-offset-2 cursor-pointer transition-all"
                />
              </div>
            )}
            
            <div className="flex-1 space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge 
                  variant="outline" 
                  className="font-semibold text-xs tracking-wide border-2 border-primary text-primary"
                >
                  {parlay.label.replace(/_/g, " ").toUpperCase()}
                </Badge>
                <Badge variant="secondary" className="font-medium">
                  {parlay.num_legs} Legs
                </Badge>
              </div>
              
              <div>
                <p className="text-4xl font-bold text-gradient tracking-tight">
                  {parlay.implied_american}
                </p>
                <p className="text-sm text-muted-foreground mt-1">American Odds</p>
              </div>
            </div>
          </div>

          <div className="flex flex-col items-end gap-3">
            <div className="text-right">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                Expected Value
              </p>
              <p className="text-3xl font-bold text-success">
                +{(parlay.ev * 100).toFixed(1)}%
              </p>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="opacity-0 group-hover:opacity-100 transition-all duration-300 hover:bg-primary/10"
            >
              {copied ? (
                <Check className="w-4 h-4 text-success" />
              ) : (
                <Copy className="w-4 h-4 text-muted-foreground" />
              )}
            </Button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className={cn(
          "grid grid-cols-2 gap-4 p-4 rounded-xl mb-4 bg-gradient-to-br",
          getEVGradient(parlay.ev)
        )}>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-background/80 backdrop-blur-sm">
              <Target className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Hit Probability
              </p>
              <p className="text-xl font-bold text-foreground">
                {(parlay.hit_probability * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-background/80 backdrop-blur-sm">
              <Trophy className="w-5 h-5 text-accent" />
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Kelly Fraction
              </p>
              <p className="text-xl font-bold text-foreground">
                {(parlay.kelly_fraction * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        {/* Expand Button */}
        <Button
          variant="ghost"
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full justify-between hover:bg-secondary/80 transition-all duration-300 group/btn"
        >
          <span className="text-sm font-semibold flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            {isExpanded ? "Hide" : "Show"} Leg Details
          </span>
          <div className="transition-transform duration-300 group-hover/btn:translate-x-1">
            {isExpanded ? (
              <ChevronUp className="w-5 h-5" />
            ) : (
              <ChevronDown className="w-5 h-5" />
            )}
          </div>
        </Button>

        {/* Expanded Legs */}
        {isExpanded && (
          <div className="mt-4 space-y-3 animate-fade-in">
            {parlay.legs.map((leg, idx) => {
              const edgePercent = ((leg.model_probability - leg.implied_probability) * 100).toFixed(1);
              const hasEdge = leg.model_probability > leg.implied_probability;
              
              return (
                <div
                  key={idx}
                  className="group/leg p-5 rounded-xl bg-gradient-to-br from-secondary/50 to-muted/30 border border-border/50 hover:border-primary/30 transition-all duration-300 hover:shadow-md"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <p className="font-bold text-lg text-foreground mb-1">{leg.player}</p>
                      <p className="text-sm text-muted-foreground font-medium">{leg.team}</p>
                    </div>
                    <Badge 
                      variant="outline" 
                      className={cn(
                        "font-bold text-sm px-3 py-1",
                        leg.odds > 0 ? "border-warning text-warning" : "border-primary text-primary"
                      )}
                    >
                      {leg.odds > 0 ? "+" : ""}{leg.odds}
                    </Badge>
                  </div>

                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-background/50 backdrop-blur-sm">
                      <p className="text-sm font-semibold text-foreground mb-1">
                        {leg.proposition}
                      </p>
                      <p className="text-xs text-muted-foreground">{leg.game}</p>
                    </div>

                    <div className="grid grid-cols-3 gap-3">
                      <div className="text-center p-2 rounded-lg bg-background/30">
                        <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                          Market
                        </p>
                        <p className="text-sm font-bold text-foreground">
                          {(leg.implied_probability * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-primary/10">
                        <p className="text-xs text-primary mb-1 uppercase tracking-wide font-medium">
                          Model
                        </p>
                        <p className="text-sm font-bold text-primary">
                          {(leg.model_probability * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="text-center p-2 rounded-lg bg-background/30">
                        <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wide">
                          Edge
                        </p>
                        <p className={cn(
                          "text-sm font-bold",
                          hasEdge ? "text-success" : "text-muted-foreground"
                        )}>
                          {hasEdge ? "+" : ""}{edgePercent}%
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </Card>
  );
};
