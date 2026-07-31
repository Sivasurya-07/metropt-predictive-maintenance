"use client";

import { useTelemetryStore } from "@/store/useStore";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { BarChart2 } from "lucide-react";

export function ShapBarChart() {
  const topFeatures = useTelemetryStore((state) => state.topFeatures);

  // Format data for Recharts: [{ name: 'Oil Temperature', value: 0.25 }]
  const data = topFeatures.map((f: any) => {
    const key = Object.keys(f)[0];
    return {
      name: key.replace(/_/g, ' '), // Make names readable
      value: Math.abs(f[key] * 100), // Convert to percentage
    };
  }).sort((a, b) => b.value - a.value); // Sort descending

  if (data.length === 0) {
    return (
      <Card className="border-border/50 shadow-sm bg-card/50">
        <CardHeader className="pb-2">
          <CardTitle className="text-xl flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-primary" />
            Top AI Contributors
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground flex items-center justify-center h-48 bg-secondary/30 rounded-xl border border-dashed border-border">
            Waiting for SHAP analysis...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border/50 shadow-sm bg-card/50 h-full">
      <CardHeader className="pb-0">
        <CardTitle className="text-xl flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-primary" />
          Top AI Contributors
        </CardTitle>
        <CardDescription className="text-xs">
          The physical sensors driving the AI's current risk assessment.
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-4 h-[220px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 0, right: 30, left: 0, bottom: 0 }}
          >
            <XAxis type="number" hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12, fontWeight: 500 }}
              width={120}
            />
            <Tooltip 
              cursor={{ fill: 'transparent' }}
              contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }}
              formatter={(val: any) => [`${Number(val).toFixed(1)}%`, 'Contribution']}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
              {data.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={index === 0 ? 'hsl(var(--destructive))' : (index === 1 ? 'hsl(var(--amber-500))' : 'hsl(var(--primary))')} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
