"use client";

import { useTelemetryStore } from "@/store/useStore";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Cpu } from "lucide-react";

export function AIPredictionPanel() {
  const predictions = useTelemetryStore((state) => state.predictions);

  // If no predictions yet
  if (!predictions || predictions.length === 0) {
    return (
      <Card className="border-border/50 shadow-sm bg-card/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            AI Predictive Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground flex items-center justify-center h-48 bg-secondary/30 rounded-xl border border-dashed border-border">
            Waiting for initial AI inference...
          </div>
        </CardContent>
      </Card>
    );
  }

  // Calculate Overall Risk & Recommendation
  const currentMaxAlert = predictions.reduce((max: string, p: any) => {
    if (p.alert_level === "emergency") return "emergency";
    if (max !== "emergency" && p.alert_level === "critical") return "critical";
    if (max !== "emergency" && max !== "critical" && p.alert_level === "warning") return "warning";
    return max;
  }, "normal");

  let riskLabel = "NORMAL";
  let riskColor = "text-success";
  let recommendation = "Continue operation";

  if (currentMaxAlert === "emergency") {
    riskLabel = "EMERGENCY";
    riskColor = "text-destructive";
    recommendation = "Halt train immediately; critical failure imminent.";
  } else if (currentMaxAlert === "critical") {
    riskLabel = "HIGH";
    riskColor = "text-amber-500";
    recommendation = "Schedule maintenance immediately upon arrival.";
  } else if (currentMaxAlert === "warning") {
    riskLabel = "MODERATE";
    riskColor = "text-amber-500"; // Assuming orange in a more complex palette, fallback to amber
    recommendation = "Inspect compressor within 24 hours.";
  }

  // Use the 4h confidence as overall for display purposes
  const conf = predictions.find(p => p.horizon === '4h')?.confidence || predictions[0].confidence;

  return (
    <Card className="border-border/50 shadow-sm bg-card/50 h-full flex flex-col">
      <CardHeader className="pb-4">
        <CardTitle className="text-xl flex items-center gap-2">
          <Cpu className="w-5 h-5 text-primary" />
          AI Predictive Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col gap-6">
        
        {/* Top Section: Horizons */}
        <div className="flex flex-col gap-3">
          {predictions.map((pred) => {
            let indicator = "bg-success shadow-[0_0_8px_rgba(16,185,129,0.5)]";
            if (pred.alert_level === "emergency") indicator = "bg-destructive shadow-[0_0_8px_rgba(220,38,38,0.5)]";
            else if (pred.alert_level === "critical") indicator = "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]";
            else if (pred.alert_level === "warning") indicator = "bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.5)]";

            return (
              <div key={pred.horizon} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                <span className="font-bold text-lg">{pred.horizon.replace('h', ' Hours')}</span>
                <div className="flex items-center gap-4">
                  <span className="font-bold text-lg tabular-nums">
                    {(pred.failure_probability * 100).toFixed(0)}%
                  </span>
                  <div className={`w-4 h-4 rounded-full ${indicator}`} />
                </div>
              </div>
            );
          })}
        </div>

        <div className="w-full h-px bg-border/50" />

        {/* Bottom Section: Details */}
        <div className="flex flex-col gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Overall Risk</span>
            <span className={`font-bold ${riskColor}`}>{riskLabel}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-bold">{(conf * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between mt-2 pt-2 border-t border-border/50">
            <span className="text-muted-foreground">Recommendation</span>
            <span className="font-semibold text-right max-w-[60%]">{recommendation}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
