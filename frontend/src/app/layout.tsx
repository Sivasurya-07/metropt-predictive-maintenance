import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";
import { WebSocketProvider } from "@/components/WebSocketProvider";

const outfit = Outfit({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "MetroPT APU Predictive Maintenance",
  description: "Real-time telemetry and predictive maintenance for APU systems.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${outfit.className} bg-background text-foreground antialiased min-h-screen`}>
        <WebSocketProvider>
          {children}
        </WebSocketProvider>
      </body>
    </html>
  );
}
