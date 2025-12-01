import { TrendingUp, Target, Percent, Layers } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatsOverviewProps {
  data: {
    num_parlays: number;
    avgEV: number;
    avgHitProb: number;
    maxLegs: number;
  };
}

export const StatsOverview = ({ data }: StatsOverviewProps) => {
  const stats = [
    {
      label: "Total Parlays",
      value: data.num_parlays.toLocaleString(),
      icon: Layers,
      color: "text-primary",
      bgGradient: "from-primary/20 to-primary/5",
      iconBg: "bg-primary/10",
    },
    {
      label: "Avg Expected Value",
      value: `+${(data.avgEV * 100).toFixed(1)}%`,
      icon: TrendingUp,
      color: "text-success",
      bgGradient: "from-success/20 to-success/5",
      iconBg: "bg-success/10",
    },
    {
      label: "Avg Hit Probability",
      value: `${(data.avgHitProb * 100).toFixed(1)}%`,
      icon: Target,
      color: "text-accent",
      bgGradient: "from-accent/20 to-accent/5",
      iconBg: "bg-accent/10",
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-6">
      {stats.map((stat, index) => (
        <Card
          key={stat.label}
          className={cn(
            "group relative overflow-hidden card-glow border-border/50",
            "animate-fade-in"
          )}
          style={{ 
            animationDelay: `${index * 100}ms`,
            animationFillMode: 'backwards'
          }}
        >
          {/* Background gradient */}
          <div className={cn(
            "absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500",
            stat.bgGradient
          )} />

          <div className="relative p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                  {stat.label}
                </p>
                <p className={cn(
                  "text-4xl font-bold tracking-tight",
                  stat.color
                )}>
                  {stat.value}
                </p>
              </div>
              
              <div className={cn(
                "p-3 rounded-xl transition-all duration-300 group-hover:scale-110 group-hover:rotate-3",
                stat.iconBg
              )}>
                <stat.icon className={cn("w-6 h-6", stat.color)} />
              </div>
            </div>

            {/* Progress indicator for visual interest */}
            <div className="h-1 rounded-full bg-muted overflow-hidden">
              <div 
                className={cn(
                  "h-full rounded-full transition-all duration-1000 delay-300",
                  stat.color.replace('text-', 'bg-')
                )}
                style={{ 
                  width: index === 0 ? '100%' : index === 1 ? '85%' : index === 2 ? '70%' : '60%',
                  animationDelay: `${index * 100 + 300}ms`
                }}
              />
            </div>
          </div>
        </Card>
      ))}
    </div>
  );
};
