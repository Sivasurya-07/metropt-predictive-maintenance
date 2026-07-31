"use client";

import { useTelemetryStore } from "@/store/useStore";
import { CheckCircle2, XCircle, Zap, Server, Activity, Cpu } from "lucide-react";

export function StatusStrip() {
  const isConnected = useTelemetryStore((state) => state.isConnected);
  const latency = useTelemetryStore((state) => state.inferenceLatencyMs);

  return (
    <div className="flex flex-wrap items-center gap-3 text-xs font-medium">
      {/* Inference Latency */}
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-border/50 rounded-md text-muted-foreground shadow-sm">
        <Zap className="w-3.5 h-3.5 text-amber-500" />
        <span>Inference {latency > 0 ? `${latency}ms` : '--'}</span>
      </div>

      {/* Redis Pipeline */}
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-border/50 rounded-md text-muted-foreground shadow-sm">
        <Server className="w-3.5 h-3.5 text-blue-500" />
        <span>Redis</span>
        {isConnected ? (
          <CheckCircle2 className="w-3.5 h-3.5 text-success ml-0.5" />
        ) : (
          <XCircle className="w-3.5 h-3.5 text-destructive ml-0.5" />
        )}
      </div>

      {/* WebSocket Status */}
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-border/50 rounded-md text-muted-foreground shadow-sm">
        <Activity className="w-3.5 h-3.5 text-indigo-500" />
        <span>WebSocket</span>
        {isConnected ? (
          <span className="text-success font-bold ml-0.5">(Live)</span>
        ) : (
          <span className="text-destructive font-bold ml-0.5">(Offline)</span>
        )}
      </div>

      {/* Model Version */}
      <div className="flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-border/50 rounded-md text-muted-foreground shadow-sm">
        <Cpu className="w-3.5 h-3.5 text-purple-500" />
        <span>Model v2.1</span>
      </div>
    </div>
  );
}
