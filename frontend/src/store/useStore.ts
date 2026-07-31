import { create } from 'zustand';

export type AlertLevel = "normal" | "warning" | "critical" | "emergency";

export interface Prediction {
  horizon: string;
  failure_probability: number;
  alert_level: AlertLevel;
  confidence: number;
}

export interface TelemetryState {
  // Raw Sensor Data
  sensorData: Record<string, number>;
  historicalData: Array<Record<string, number | string>>;
  
  // ML Predictions
  predictions: Prediction[];
  topFeatures: string[];
  narrative: string;
  
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
  narrative: "Initializing system...",
  isConnected: false,
  
  setTelemetry: (data) => set((state) => {
    // Keep last 100 points for historical chart visualization
    const newHistorical = [...state.historicalData, { 
      timestamp: data.timestamp || new Date().toISOString(), 
      ...data.sensor_readings 
    }].slice(-100);
    
    return {
      sensorData: data.sensor_readings || {},
      historicalData: newHistorical,
      predictions: data.predictions || [],
      topFeatures: data.top_features || [],
      narrative: data.narrative || state.narrative,
    };
  }),
  
  setConnectionStatus: (status) => set({ isConnected: status }),
}));
