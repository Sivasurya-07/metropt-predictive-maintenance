"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Settings, MousePointerClick, Info, Activity } from "lucide-react";
import { useState, useMemo } from "react";
import { useTelemetryStore } from "@/store/useStore";

export function APUSchematic() {
  const [activeSubsystem, setActiveSubsystem] = useState<string | null>("Compressor");
  
  const sensorData = useTelemetryStore((state) => state.sensorData);
  const subsystemShap = useTelemetryStore((state) => state.subsystemShap);
  const topFeatures = useTelemetryStore((state) => state.topFeatures);
  const predictions = useTelemetryStore((state) => state.predictions);

  // Derive dynamic live data
  const currentData = useMemo(() => {
    if (!activeSubsystem || !sensorData || Object.keys(sensorData).length === 0) return null;
    
    // Overall Risk determines baseline health
    const maxRisk = predictions.length > 0 ? 
      Math.max(...predictions.map(p => p.failure_probability)) : 0;
      
    const shapImpact = subsystemShap[activeSubsystem] || 0;
    
    // Health is degraded mostly by the overall risk, weighted by this subsystem's SHAP contribution
    // e.g. If overall risk is 40%, and this subsystem is 80% of the cause, its health takes a big hit.
    const healthPenalty = (maxRisk * 100) * (shapImpact / 100);
    const healthValue = Math.max(0, 100 - healthPenalty);
    
    let temp = "N/A";
    let press = "N/A";
    
    if (activeSubsystem === "Compressor") {
      temp = sensorData.Oil_temperature ? `${sensorData.Oil_temperature.toFixed(1)}°C` : "N/A";
      press = sensorData.TP2 ? `${sensorData.TP2.toFixed(1)} bar` : "N/A";
    } else if (activeSubsystem === "Reservoir") {
      temp = "N/A"; // No dedicated temp sensor for reservoir in this dataset
      press = sensorData.TP3 ? `${sensorData.TP3.toFixed(1)} bar` : "N/A";
    } else if (activeSubsystem === "Motor") {
      temp = "N/A"; 
      press = sensorData.Motor_current ? `${sensorData.Motor_current.toFixed(1)} A` : "N/A";
    } else if (activeSubsystem === "Valves") {
      temp = sensorData.H1 ? `${sensorData.H1.toFixed(1)}°C` : "N/A"; // Using H1 as proxy for valve area temp
      press = sensorData.DV_pressure ? `${sensorData.DV_pressure.toFixed(2)} bar` : "N/A";
    }

    // Find the primary reason (top feature belonging to this subsystem)
    let reason = "Normal operation";
    if (topFeatures.length > 0 && shapImpact > 10) {
        // Simple heuristic: just show the very top feature overall if it's impactful
        const topFeatureObj = topFeatures[0];
        const topFeatureName = Object.keys(topFeatureObj)[0];
        reason = topFeatureName.replace(/_/g, ' ');
    }

    return {
      health: `${healthValue.toFixed(1)}%`,
      healthRaw: healthValue,
      temp,
      press,
      shap: `+${shapImpact.toFixed(1)}%`,
      reason
    };
  }, [activeSubsystem, sensorData, subsystemShap, topFeatures, predictions]);


  // Helper to determine active subsystem color intensity based on SHAP
  const getSubsystemStyle = (name: string) => {
    const shap = subsystemShap[name] || 0;
    const isActive = activeSubsystem === name;
    
    let baseBorder = "border-border";
    let baseBg = "bg-card/50";
    
    if (shap > 50) {
      baseBorder = "border-destructive";
      baseBg = "bg-destructive/10";
    } else if (shap > 20) {
      baseBorder = "border-amber-500";
      baseBg = "bg-amber-500/10";
    }

    if (isActive) {
      if (shap > 50) return "border-destructive bg-destructive/30 ring-2 ring-destructive/50";
      if (shap > 20) return "border-amber-500 bg-amber-500/30 ring-2 ring-amber-500/50";
      return "border-primary bg-primary/20 ring-2 ring-primary/50";
    }
    
    return `${baseBorder} ${baseBg} hover:border-primary/50`;
  };

  return (
    <Card className="border-border/50 shadow-sm bg-card/50 overflow-hidden relative">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" />
            Interactive APU Schematic
          </div>
          <div className="text-xs flex items-center gap-1.5 text-muted-foreground bg-secondary/50 px-2 py-1 rounded border border-border/50">
            <Activity className="w-3.5 h-3.5 text-success animate-pulse" />
            Live Data Connected
          </div>
        </CardTitle>
        <CardDescription className="text-sm">
          Click physical subsystems to inspect real-time sensor metrics and localized AI failure attribution.
        </CardDescription>
      </CardHeader>
      
      <CardContent className="p-0 flex flex-col md:flex-row min-h-[300px]">
        {/* Left Side: Schematic Diagram */}
        <div className="flex-1 bg-secondary/20 p-6 flex items-center justify-center relative border-r border-border/50 min-h-[250px]">
          {/* Vector Art using HTML shapes */}
          <div className="relative w-full max-w-[400px] aspect-[4/3]">
            
            {/* Reservoir */}
            <div 
              onClick={() => setActiveSubsystem("Reservoir")}
              className={`absolute top-[10%] left-[10%] w-[80%] h-[20%] border-2 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300 ${getSubsystemStyle("Reservoir")}`}
            >
              <span className="font-bold text-sm">Reservoir</span>
            </div>
            
            {/* Compressor */}
            <div 
              onClick={() => setActiveSubsystem("Compressor")}
              className={`absolute top-[40%] left-[30%] w-[40%] h-[30%] border-2 flex items-center justify-center cursor-pointer transition-all duration-300 ${getSubsystemStyle("Compressor")}`}
            >
              <span className="font-bold text-sm">Compressor</span>
            </div>

            {/* Motor */}
            <div 
              onClick={() => setActiveSubsystem("Motor")}
              className={`absolute top-[45%] left-[5%] w-[20%] h-[20%] border-2 rounded-sm flex items-center justify-center cursor-pointer transition-all duration-300 ${getSubsystemStyle("Motor")}`}
            >
              <span className="font-bold text-sm">Motor</span>
            </div>

            {/* Valves */}
            <div 
              onClick={() => setActiveSubsystem("Valves")}
              className={`absolute top-[75%] left-[45%] w-[10%] h-[15%] border-2 flex items-center justify-center cursor-pointer transition-all duration-300 ${getSubsystemStyle("Valves")}`}
            >
              <span className="font-bold text-xs rotate-90">Valves</span>
            </div>

            {/* Connecting Pipes */}
            <div className="absolute top-[30%] left-[50%] w-0.5 h-[10%] bg-border" />
            <div className="absolute top-[55%] left-[25%] w-[5%] h-0.5 bg-border" />
            <div className="absolute top-[70%] left-[50%] w-0.5 h-[5%] bg-border" />
          </div>
        </div>

        {/* Right Side: Detail Panel */}
        <div className="w-full md:w-[280px] p-6 bg-card/50 flex flex-col gap-4">
          {!currentData ? (
            <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground text-center gap-2">
              <Info className="w-8 h-8 opacity-50" />
              <p className="text-sm">Waiting for telemetry data stream...</p>
            </div>
          ) : (
            <>
              <div className="border-b border-border/50 pb-3">
                <h3 className="text-xl font-bold tracking-tight text-foreground">{activeSubsystem}</h3>
              </div>
              <div className="flex flex-col gap-3 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Health Status</span>
                  <span className={`font-bold ${currentData.healthRaw < 80 ? (currentData.healthRaw < 50 ? 'text-destructive' : 'text-amber-500') : 'text-success'}`}>
                    {currentData.health}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Temperature</span>
                  <span className="font-semibold text-foreground tabular-nums">{currentData.temp}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Pressure/Current</span>
                  <span className="font-semibold text-foreground tabular-nums">{currentData.press}</span>
                </div>
                
                <div className="mt-2 pt-3 border-t border-border/50 flex flex-col gap-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">SHAP Impact</span>
                    <span className="font-bold text-destructive bg-destructive/10 px-2 py-0.5 rounded">{currentData.shap}</span>
                  </div>
                  <div className="flex flex-col gap-1 mt-1">
                    <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Primary Reason</span>
                    <span className="text-sm font-medium">{currentData.reason}</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
