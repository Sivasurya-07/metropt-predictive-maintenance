"use client";

import { useTelemetryStore } from "@/store/useStore";
import { AlertBanner } from "@/components/AlertBanner";
import { TelemetryChart } from "@/components/TelemetryChart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Activity, Gauge, Thermometer, Zap, ShieldAlert, Cpu } from "lucide-react";

export default function Dashboard() {
  const sensorData = useTelemetryStore((state) => state.sensorData);
  const predictions = useTelemetryStore((state) => state.predictions);
  const topFeatures = useTelemetryStore((state) => state.topFeatures);
  const narrative = useTelemetryStore((state) => state.narrative);

  return (
    <main className="w-full p-4 md:px-12 md:py-8 flex flex-col gap-10">
      
      {/* Header Section */}
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-primary/10 rounded-xl">
            <ShieldAlert className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight">
              MetroPT APU Predictive Maintenance
            </h1>
            <p className="text-muted-foreground mt-1 text-lg max-w-3xl">
              This dashboard monitors the Air Production Unit (APU) on Metro trains in real-time. It uses a 
              Machine Learning AI to predict mechanical failures hours before they happen, ensuring passenger 
              safety and reducing train downtime.
            </p>
          </div>
        </div>
      </header>

      <AlertBanner />

      {/* High-Level System Status Metrics */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Gauge className="w-5 h-5 text-muted-foreground" />
            Current System State
          </h2>
          <p className="text-muted-foreground text-sm">
            Instantaneous readings from the APU's core physical sensors.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard 
            title="Reservoir Pressure" 
            value={sensorData.Reservoirs ? `${sensorData.Reservoirs.toFixed(2)} bar` : "--"} 
            icon={<Gauge className="w-5 h-5 text-primary" />} 
          />
          <MetricCard 
            title="Motor Current" 
            value={sensorData.Motor_current ? `${sensorData.Motor_current.toFixed(2)} A` : "--"} 
            icon={<Zap className="w-5 h-5 text-amber-500" />} 
          />
          <MetricCard 
            title="Oil Temperature" 
            value={sensorData.Oil_temperature ? `${sensorData.Oil_temperature.toFixed(2)} °C` : "--"} 
            icon={<Thermometer className="w-5 h-5 text-destructive" />} 
          />
          <MetricCard 
            title="Compressor Status" 
            value={sensorData.COMP === 1 ? "ACTIVE" : (sensorData.COMP === 0 ? "STANDBY" : "--")} 
            icon={<Activity className="w-5 h-5 text-success" />} 
          />
        </div>
      </section>

      {/* Main Analytical Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Live Telemetry Chart */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <Card className="h-full border-border/50 bg-card/50 shadow-sm">
            <CardHeader>
              <CardTitle className="text-xl">Live Sensor Telemetry</CardTitle>
              <CardDescription className="text-sm">
                These sensors measure the pressure (bar) inside the APU's core components. By analyzing these tiny 
                fluctuations over time, our AI can detect the earliest invisible signs of wear and tear.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TelemetryChart />
              
              {/* Telemetry Legend / Explanation */}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 border-t border-border/50 pt-5">
                
                <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-card border border-border/50 shadow-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3.5 h-3.5 rounded-full bg-[#10b981] shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                    <span className="font-bold text-sm text-foreground">Compressor Pressure</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <strong>The "Heartbeat".</strong> Shows how hard the main engine is working to compress and pump air into the system.
                  </p>
                </div>

                <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-card border border-border/50 shadow-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3.5 h-3.5 rounded-full bg-[#3b82f6] shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                    <span className="font-bold text-sm text-foreground">Reservoir Pressure</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <strong>The "Lungs".</strong> Measures how much compressed air is safely stored and ready for the train brakes.
                  </p>
                </div>

                <div className="flex flex-col gap-1.5 p-3 rounded-xl bg-card border border-border/50 shadow-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3.5 h-3.5 rounded-full bg-[#8b5cf6] shadow-[0_0_8px_rgba(139,92,246,0.5)]" />
                    <span className="font-bold text-sm text-foreground">Valve Temperature</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <strong>The "Thermometer".</strong> Tracks heat inside the system to ensure the machinery doesn't dangerously overheat.
                  </p>
                </div>

              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: AI Predictions & Narrative */}
        <div className="col-span-1 flex flex-col gap-6">
          
          <Card className="border-border/50 shadow-sm bg-card/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-xl flex items-center gap-2">
                <Cpu className="w-5 h-5 text-primary" />
                AI Failure Prediction
              </CardTitle>
              <CardDescription className="text-sm">
                Our Machine Learning model looks at the sensor data to predict the likelihood of a breakdown 
                within the next 2, 4, and 8 hours. <strong>Green</strong> means healthy, <strong>Red</strong> means danger.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              {predictions.length > 0 ? (
                predictions.map((pred) => (
                  <div key={pred.horizon} className="flex flex-col gap-2 p-4 rounded-xl bg-background border border-border/50">
                    <div className="flex justify-between items-end">
                      <span className="font-semibold text-foreground">In next {pred.horizon}</span>
                      <span className="text-sm font-bold">
                        {(pred.failure_probability * 100).toFixed(1)}% Risk
                      </span>
                    </div>
                    <div className="w-full bg-secondary h-2.5 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-1000 ease-out ${
                          pred.failure_probability > 0.5 ? 'bg-destructive' : 
                          pred.failure_probability > 0.2 ? 'bg-amber-500' : 'bg-success'
                        }`}
                        style={{ width: `${Math.max(2, pred.failure_probability * 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>Model Confidence: {(pred.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-sm text-muted-foreground flex items-center justify-center h-32 bg-secondary/30 rounded-xl border border-dashed border-border">
                  Waiting for initial AI inference...
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-border/50 shadow-sm bg-primary/5 border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">AI Diagnosis Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-relaxed text-foreground/90 font-medium">
                {narrative}
              </p>
              {topFeatures.length > 0 && (
                <div className="mt-4 pt-4 border-t border-primary/10">
                  <h4 className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">
                    Key Influencing Sensors
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {topFeatures.map((f: any) => {
                      const key = Object.keys(f)[0];
                      return (
                        <span key={key} className="text-xs bg-primary/10 border border-primary/20 px-2 py-1 rounded-md text-primary font-medium">
                          {key.replace(/_/g, ' ')}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

        </div>
      </div>
    </main>
  );
}

function MetricCard({ title, value, icon }: { title: string, value: string, icon: React.ReactNode }) {
  return (
    <Card className="border-border/50 shadow-sm bg-card/50 transition-all hover:bg-card">
      <CardContent className="p-6 flex flex-row items-center justify-between space-y-0">
        <div className="flex flex-col gap-1.5">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold tracking-tight text-foreground">{value}</p>
        </div>
        <div className="p-3 bg-secondary rounded-xl border border-border/50 shadow-inner">
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}
