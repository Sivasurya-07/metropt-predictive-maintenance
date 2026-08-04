"use client";

import { useEffect, useRef } from "react";
import { useTelemetryStore } from "@/store/useStore";

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const setTelemetry = useTelemetryStore((state) => state.setTelemetry);
  const setConnectionStatus = useTelemetryStore((state) => state.setConnectionStatus);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Determine the WebSocket URL for the Railway backend
    const RAILWAY_BACKEND = "wss://web-production-c421a.up.railway.app/ws/alerts";
    let wsUrl = RAILWAY_BACKEND;
    if (process.env.NEXT_PUBLIC_WS_URL) {
      wsUrl = process.env.NEXT_PUBLIC_WS_URL;
    } else if (process.env.NEXT_PUBLIC_API_URL) {
      try {
        const url = new URL(process.env.NEXT_PUBLIC_API_URL);
        const protocol = (url.protocol === "https:" || url.protocol === "wss:") ? "wss:" : "ws:";
        wsUrl = `${protocol}//${url.host}/ws/alerts`;
      } catch (e) {
        console.error("Failed to parse NEXT_PUBLIC_API_URL:", e);
      }
    }
    
    // Initial fetch to populate UI before live stream arrives
    const httpUrl = "https://web-production-c421a.up.railway.app/latest";
    fetch(httpUrl)
      .then((res) => res.json())
      .then((initialData) => {
        if (initialData && initialData.sensor_readings) {
          setTelemetry(initialData);
        }
      })
      .catch((err) => console.warn("[WebSocket] Failed initial /latest fetch:", err));

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      let pingInterval: any = null;

      ws.onopen = () => {
        console.log("[WebSocket] Connected to APU Telemetry Stream");
        setConnectionStatus(true);
        // Send a ping every 15 seconds to keep Railway connection alive
        pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 15000);
      };

      ws.onmessage = (event) => {
        try {
          if (event.data === "pong") return; // Ignore ping/pong keepalives
          const data = JSON.parse(event.data);
          setTelemetry(data);
        } catch (error) {
          console.error("Failed to parse telemetry:", error);
        }
      };

      ws.onclose = () => {
        if (pingInterval) clearInterval(pingInterval);
        console.warn("[WebSocket] Disconnected. Reconnecting in 3s...");
        setConnectionStatus(false);
        setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        if (pingInterval) clearInterval(pingInterval);
        console.error("[WebSocket] Error:", error);
        ws.close();
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [setTelemetry, setConnectionStatus]);

  return <>{children}</>;
}
