"use client";

import { useTelemetryStore } from "@/store/useStore";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format, isValid } from "date-fns";

function safeFormatTime(timestamp: any): string {
  if (!timestamp) return "--:--:--";
  try {
    const date = new Date(timestamp);
    if (!isValid(date)) return "--:--:--";
    return format(date, "HH:mm:ss");
  } catch {
    return "--:--:--";
  }
}

export function TelemetryChart() {
  const history = useTelemetryStore((state) => state.historicalData);

  if (history.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground bg-muted/20 rounded-xl border border-border border-dashed">
        Awaiting telemetry data...
      </div>
    );
  }

  // Map history to recharts data format safely filtering invalid points
  const data = history
    .map((point) => {
      const ts = point.timestamp ? new Date(point.timestamp).getTime() : NaN;
      return {
        time: ts,
        TP2: typeof point.TP2 === "number" ? point.TP2 : 0,
        TP3: typeof point.TP3 === "number" ? point.TP3 : 0,
        H1: typeof point.H1 === "number" ? point.H1 : 0,
      };
    })
    .filter((point) => !isNaN(point.time));

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground bg-muted/20 rounded-xl border border-border border-dashed">
        Processing telemetry timestamps...
      </div>
    );
  }

  return (
    <div className="h-[350px] w-full mt-4">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
          <XAxis 
            dataKey="time" 
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(unixTime) => safeFormatTime(unixTime)}
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickMargin={10}
            minTickGap={30}
          />
          <YAxis 
            stroke="hsl(var(--muted-foreground))" 
            fontSize={12} 
            tickMargin={10}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{ 
              backgroundColor: "hsl(var(--card))", 
              borderColor: "hsl(var(--border))",
              borderRadius: "8px",
              color: "hsl(var(--foreground))"
            }}
            labelFormatter={(label: any) => safeFormatTime(Number(label))}
            itemStyle={{ fontSize: 13 }}
            labelStyle={{ fontSize: 13, marginBottom: "4px", color: "hsl(var(--muted-foreground))" }}
          />
          <Line
            type="monotone"
            dataKey="TP2"
            name="Pressure TP2"
            stroke="#10b981"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="TP3"
            name="Pressure TP3"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="H1"
            name="Pressure H1"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
