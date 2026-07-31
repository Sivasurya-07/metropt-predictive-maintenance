"use client";

import { useTelemetryStore } from "@/store/useStore";
import { AlertBanner } from "@/components/AlertBanner";
import { TelemetryChart } from "@/components/TelemetryChart";
import { SensorGauges } from "@/components/SensorGauges";
import { AIPredictionPanel } from "@/components/AIPredictionPanel";
import { ShapBarChart } from "@/components/ShapBarChart";
import { RiskSparkline } from "@/components/RiskSparkline";
import { EventTimeline } from "@/components/EventTimeline";
import { APUSchematic } from "@/components/APUSchematic";
import { StatusStrip } from "@/components/StatusStrip";
import { ShieldAlert } from "lucide-react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";

export default function Dashboard() {
  const latency = useTelemetryStore((state) => state.inferenceLatencyMs);

  return (
    <main className="w-full p-4 md:px-12 md:py-8 flex flex-col gap-8 max-w-[1600px] mx-auto">
      
      {/* 1. Header: Title | Status */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-primary/10 rounded-xl">
            <ShieldAlert className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
              MetroPT APU Predictive Maintenance
            </h1>
            <p className="text-muted-foreground mt-1 text-sm md:text-base max-w-2xl">
              Real-time monitoring and AI diagnostics for railway Air Production Units.
            </p>
          </div>
        </div>
        <StatusStrip />
      </header>

      <AlertBanner />

      {/* 2. Sensor Gauges */}
      <SensorGauges />

      {/* Main Analytical Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column (Span 2) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* 3. Interactive APU Schematic */}
          <APUSchematic />
          
          {/* 4. Live Telemetry */}
          <Card className="border-border/50 shadow-sm bg-card/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">Live Sensor Telemetry</CardTitle>
              <CardDescription className="text-sm">
                Raw time-series data from the edge device.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TelemetryChart />
            </CardContent>
          </Card>
        </div>

        {/* Right Column (Span 1) */}
        <div className="col-span-1 flex flex-col gap-6">
          {/* 5. AI Predictive Analysis */}
          <AIPredictionPanel />
          
          {/* 6. SHAP Top Contributors */}
          <div className="h-[300px]">
            <ShapBarChart />
          </div>

          {/* 7. Risk Trend */}
          <RiskSparkline />
          
          {/* 8. Event Timeline */}
          <div className="h-[250px]">
            <EventTimeline />
          </div>
        </div>
      </div>

      {/* 9. Footer: Dataset/Model/Inference badge */}
      <footer className="mt-8 pt-4 border-t border-border/50 text-center flex justify-center">
        <div className="inline-flex items-center gap-2 text-xs text-muted-foreground bg-secondary/30 px-3 py-1.5 rounded-full border border-border/50">
          <span>MetroPT Dataset</span>
          <span>&middot;</span>
          <span>Stacked Ensemble (LightGBM + XGBoost + CNN)</span>
          <span>&middot;</span>
          <span>Inference {latency > 0 ? latency : 27}ms</span>
          <span>&middot;</span>
          <span>Last Update {new Date().toLocaleTimeString()}</span>
        </div>
      </footer>
      
    </main>
  );
}
