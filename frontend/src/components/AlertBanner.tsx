"use client";

import { useTelemetryStore } from "@/store/useStore";
import { AlertTriangle, CheckCircle2, ShieldAlert, Loader2 } from "lucide-react";

export function AlertBanner() {
  const predictions = useTelemetryStore((state) => state.predictions);
  const narrative = useTelemetryStore((state) => state.narrative);
  const isConnected = useTelemetryStore((state) => state.isConnected);

  if (!isConnected) {
    return (
      <div className="w-full rounded-2xl p-6 bg-secondary border border-border flex items-center justify-center gap-3 text-muted-foreground shadow-sm">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span className="font-medium">Connecting to APU telemetry stream...</span>
      </div>
    );
  }

  // Find the highest alert level based on our 3 horizons
  const hasEmergency = predictions.some(p => p.alert_level === "emergency");
  const hasCritical = predictions.some(p => p.alert_level === "critical");
  const hasWarning = predictions.some(p => p.alert_level === "warning");

  if (hasEmergency) {
    return (
      <div className="w-full rounded-2xl p-6 bg-destructive/10 border border-destructive/30 text-destructive-foreground flex flex-col md:flex-row items-center gap-5 shadow-[0_0_40px_rgba(220,38,38,0.15)] transition-all">
        <div className="p-4 bg-destructive/20 rounded-full flex-shrink-0">
          <ShieldAlert className="w-10 h-10 text-destructive animate-pulse" />
        </div>
        <div className="flex-1">
          <h2 className="text-xl font-extrabold text-destructive tracking-wide uppercase">Emergency Action Required</h2>
          <p className="text-sm mt-1.5 text-foreground/80 font-medium">{narrative}</p>
        </div>
      </div>
    );
  }

  if (hasCritical || hasWarning) {
    return (
      <div className="w-full rounded-2xl p-6 bg-amber-500/10 border border-amber-500/30 text-amber-500 flex flex-col md:flex-row items-center gap-5 shadow-[0_0_40px_rgba(245,158,11,0.1)] transition-all">
        <div className="p-4 bg-amber-500/20 rounded-full flex-shrink-0">
          <AlertTriangle className="w-10 h-10 text-amber-500" />
        </div>
        <div className="flex-1">
          <h2 className="text-xl font-extrabold text-amber-500 tracking-wide uppercase">
            {hasCritical ? "Inspect immediately — High Priority" : "Inspect compressor — Medium Priority"}
          </h2>
          <p className="text-sm mt-1.5 text-foreground/80 font-medium">{narrative}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full rounded-2xl p-6 bg-success/10 border border-success/30 flex flex-col md:flex-row items-center gap-5 transition-all shadow-[0_0_30px_rgba(16,185,129,0.05)]">
      <div className="p-4 bg-success/20 rounded-full flex-shrink-0">
        <CheckCircle2 className="w-10 h-10 text-success" />
      </div>
      <div className="flex-1">
        <h2 className="text-xl font-extrabold text-success tracking-wide uppercase">Continue Operation</h2>
        <p className="text-sm mt-1.5 text-foreground/80 font-medium">All telemetry nominal. No immediate maintenance required.</p>
      </div>
    </div>
  );
}
