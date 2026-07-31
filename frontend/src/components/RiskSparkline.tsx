"use client";

import { useTelemetryStore } from "@/store/useStore";
import { LineChart, Line, ResponsiveContainer, YAxis } from "recharts";
import { TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function RiskSparkline() {
  const riskTrend = useTelemetryStore((state) => state.riskTrend);

  // Convert array of numbers to Recharts format [{ value: 10 }, { value: 12 }]
  const data = riskTrend.map((val) => ({ value: val }));
  
  const currentRisk = riskTrend.length > 0 ? riskTrend[riskTrend.length - 1] : 0;
  let color = "hsl(var(--success))";
  if (currentRisk > 50) color = "hsl(var(--destructive))";
  else if (currentRisk > 20) color = "hsl(var(--amber-500))";

  return (
    <Card className="border-border/50 shadow-sm bg-card/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold flex items-center justify-between text-muted-foreground">
          <div className="flex items-center gap-1.5">
            <TrendingUp className="w-4 h-4" />
            4H Risk Trend
          </div>
          <span className="font-bold text-foreground" style={{ color }}>
            {currentRisk.toFixed(1)}%
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="h-[60px] pb-4 px-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <YAxis domain={[0, 100]} hide />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke={color} 
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
