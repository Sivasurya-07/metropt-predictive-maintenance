"use client";

import { useTelemetryStore } from "@/store/useStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Clock } from "lucide-react";

export function EventTimeline() {
  const events = useTelemetryStore((state) => state.events);

  return (
    <Card className="border-border/50 shadow-sm bg-card/50 h-full">
      <CardHeader className="pb-4">
        <CardTitle className="text-sm font-semibold flex items-center gap-2 text-muted-foreground">
          <Clock className="w-4 h-4" />
          Event Timeline
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-0 relative">
        {events.slice().reverse().map((event, index) => {
          let dotColor = "bg-primary";
          if (event.type === "emergency") dotColor = "bg-destructive";
          else if (event.type === "critical") dotColor = "bg-amber-500";
          else if (event.type === "warning") dotColor = "bg-orange-500";
          else if (event.type === "info") dotColor = "bg-success";

          const timeString = new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

          return (
            <div key={index} className="flex gap-4 relative pb-4">
              {/* Vertical line connecting dots, except for the last item */}
              {index !== events.length - 1 && (
                <div className="absolute left-[5px] top-4 bottom-[-16px] w-px bg-border/50" />
              )}
              
              <div className="mt-1.5 flex-none relative z-10">
                <div className={`w-3 h-3 rounded-full ${dotColor} shadow-sm border-2 border-card`} />
              </div>
              
              <div className="flex flex-col">
                <span className="text-xs font-mono text-muted-foreground">{timeString}</span>
                <span className="text-sm font-medium text-foreground">{event.message}</span>
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
