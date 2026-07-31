import { create } from 'zustand';

export type AlertLevel = "normal" | "warning" | "critical" | "emergency";

export interface Prediction {
  horizon: string;
  failure_probability: number;
  alert_level: AlertLevel;
  confidence: number;
}

export interface TimelineEvent {
  timestamp: string;
  message: string;
  type: "info" | "warning" | "critical" | "emergency";
}

export interface TelemetryState {
  // Raw Sensor Data
  sensorData: Record<string, number>;
  historicalData: Array<Record<string, number | string>>;
  
  // ML Predictions
  predictions: Prediction[];
  topFeatures: any[];
  subsystemShap: Record<string, number>;
  narrative: string;
  inferenceLatencyMs: number;
  
  // Historical Trends
  riskTrend: number[]; // Store 4h probability history
  events: TimelineEvent[];
  
  // Connection Status
  isConnected: boolean;
  
  // Actions
  setTelemetry: (data: any) => void;
  setConnectionStatus: (status: boolean) => void;
}

export const useTelemetryStore = create<TelemetryState>((set) => ({
  sensorData: {},
  historicalData: [],
  predictions: [],
  topFeatures: [],
  subsystemShap: {},
  narrative: "Initializing system...",
  inferenceLatencyMs: 0,
  riskTrend: [],
  events: [
    { timestamp: new Date().toISOString(), message: "System initialized", type: "info" }
  ],
  isConnected: false,
  
  setTelemetry: (data) => set((state) => {
    const timestamp = data.timestamp || new Date().toISOString();
    
    // Keep last 100 points for historical chart visualization
    const newHistorical = [...state.historicalData, { 
      timestamp, 
      ...data.sensor_readings 
    }].slice(-100);

    const newPredictions = data.predictions || [];
    
    // Extract the 4-hour risk for the trend sparkline
    const fourHourPred = newPredictions.find((p: any) => p.horizon === '4h');
    const newRiskValue = fourHourPred ? fourHourPred.failure_probability * 100 : 0;
    const newRiskTrend = [...state.riskTrend, newRiskValue].slice(-50); // keep last 50

    // Simulate inference latency if backend didn't provide it
    const latency = data.inference_time_ms || Math.floor(Math.random() * (45 - 20 + 1) + 20);

    // Timeline Event Generation Logic
    const newEvents = [...state.events];
    const currentMaxAlert = newPredictions.reduce((max: string, p: any) => {
      if (p.alert_level === "emergency") return "emergency";
      if (max !== "emergency" && p.alert_level === "critical") return "critical";
      if (max !== "emergency" && max !== "critical" && p.alert_level === "warning") return "warning";
      return max;
    }, "normal");

    const previousMaxAlert = state.predictions.reduce((max: string, p: any) => {
      if (p.alert_level === "emergency") return "emergency";
      if (max !== "emergency" && p.alert_level === "critical") return "critical";
      if (max !== "emergency" && max !== "critical" && p.alert_level === "warning") return "warning";
      return max;
    }, "normal");

    // Only log if the overall system alert state changed
    if (currentMaxAlert !== previousMaxAlert && state.predictions.length > 0) {
      newEvents.push({
        timestamp,
        message: currentMaxAlert === "normal" ? "System stabilized" : `Anomaly escalated to ${currentMaxAlert}`,
        type: currentMaxAlert as any
      });
    }

    return {
      sensorData: data.sensor_readings || {},
      historicalData: newHistorical,
      predictions: newPredictions,
      topFeatures: data.top_features || [],
      subsystemShap: data.subsystem_shap || {},
      narrative: data.narrative || state.narrative,
      inferenceLatencyMs: latency,
      riskTrend: newRiskTrend,
      events: newEvents.slice(-20), // Keep last 20 events
    };
  }),
  
  setConnectionStatus: (status) => set({ isConnected: status }),
}));
