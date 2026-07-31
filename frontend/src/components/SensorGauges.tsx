"use client";

import { useTelemetryStore } from "@/store/useStore";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { Card, CardContent } from "@/components/ui/card";
import { Gauge, Thermometer, Zap } from "lucide-react";

interface GaugeProps {
  title: string;
  value: number;
  min: number;
  max: number;
  unit: string;
  icon: React.ReactNode;
  color: string;
}

function CircularGauge({ title, value, min, max, unit, icon, color }: GaugeProps) {
  // Calculate percentage for the gauge
  const percentage = Math.min(Math.max((value - min) / (max - min), 0), 1);
  const data = [
    { name: "value", value: percentage },
    { name: "empty", value: 1 - percentage },
  ];

  return (
    <Card className="border-border/50 shadow-sm bg-card/50">
      <CardContent className="p-4 flex flex-col items-center justify-center relative h-[180px]">
        <div className="absolute top-4 left-4 flex items-center gap-2 text-muted-foreground">
          {icon}
          <span className="text-sm font-semibold tracking-wide uppercase">{title}</span>
        </div>
        
        <div className="w-full h-full mt-6">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="85%" // Shift down to make room for half-circle
                startAngle={180}
                endAngle={0}
                innerRadius={60}
                outerRadius={80}
                dataKey="value"
                stroke="none"
                isAnimationActive={false} // Disable animation for live streaming performance
              >
                <Cell key="cell-0" fill={color} />
                <Cell key="cell-1" fill="hsl(var(--secondary))" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        
        <div className="absolute bottom-4 flex flex-col items-center">
          <span className="text-3xl font-black tracking-tighter text-foreground">
            {value.toFixed(1)}
          </span>
          <span className="text-xs text-muted-foreground font-bold">{unit}</span>
        </div>
      </CardContent>
    </Card>
  );
}

export function SensorGauges() {
  const sensorData = useTelemetryStore((state) => state.sensorData);

  return (
    <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <CircularGauge
        title="Pressure"
        value={sensorData.Reservoirs || 0}
        min={0}
        max={12}
        unit="BAR"
        icon={<Gauge className="w-4 h-4 text-[#3b82f6]" />}
        color="#3b82f6"
      />
      <CircularGauge
        title="Temperature"
        value={sensorData.Oil_temperature || 0}
        min={20}
        max={100}
        unit="°C"
        icon={<Thermometer className="w-4 h-4 text-destructive" />}
        color="hsl(var(--destructive))"
      />
      <CircularGauge
        title="Motor Current"
        value={sensorData.Motor_current || 0}
        min={0}
        max={15}
        unit="AMPS"
        icon={<Zap className="w-4 h-4 text-amber-500" />}
        color="#f59e0b"
      />
    </section>
  );
}
