import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function TrafficDashboard() {
  const [url, setUrl] = useState("");
  const [counts, setCounts] = useState({ left: 0, straight: 0, right: 0 });
  const [videoLoaded, setVideoLoaded] = useState(false);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8765"); // Backend WebSocket 連線
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setCounts(data);
    };
    return () => ws.close();
  }, []);

  const handleStart = () => {
    fetch("http://localhost:8000/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }).then(() => setVideoLoaded(true));
  };

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-2xl font-bold">交通方向統計儀表板</h1>
      <Card>
        <CardContent className="space-y-4 p-6">
          <div className="flex items-center gap-4">
            <Input
              placeholder="請輸入即時影像網址..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <Button onClick={handleStart}>啟動分析</Button>
          </div>
          {videoLoaded && (
            <div className="mt-4">
              <iframe
                src={url}
                title="Live Stream"
                className="w-full h-[360px] border rounded-xl"
              ></iframe>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-6">
          <h2 className="text-xl font-semibold mb-4">方向統計圖</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={Object.entries(counts).map(([k, v]) => ({ name: k, count: v }))}>
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
