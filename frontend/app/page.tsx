"use client";

import React, { useState, useEffect, FormEvent } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL!;

export default function Home() {
  const [youtubeLink, setYoutubeLink] = useState("");
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStage, setLoadingStage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsProcessing(true);
    setPrediction(null);
    setLoadingProgress(0);
    setLoadingStage("");
    setErrorMessage(null);
  
    try {
      const response = await fetch(`${BACKEND_URL}/start_predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: youtubeLink }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
      }
  
      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let partialData = '';
  
      const processStream = async () => {
        if (!reader) return;
        const { done, value } = await reader.read();
        if (done) {
          setIsProcessing(false);
          return;
        }
        partialData += decoder.decode(value, { stream: true });
        const lines = partialData.split('\n\n');
        partialData = lines.pop() || ''; // Keep the last partial line
  
        lines.forEach(line => {
          if (line.startsWith('data:')) {
            const jsonData = line.substring(5).trim();
            if (jsonData === 'Simple text test') {
              console.log("Received simple text!");
              return;
            }
            try {
              const data = JSON.parse(jsonData);
              console.log("Received SSE data:", data);
              if (data.progress) {
                setLoadingProgress(data.progress);
              }
              if (data.stage) {
                setLoadingStage(data.stage);
              }
              if (data.score !== undefined) {
                setPrediction(data.score);
                setIsProcessing(false);
              }
              if (data.error) {
                setErrorMessage(data.error);
                setIsProcessing(false);
              }
            } catch (e) {
              console.error("Error parsing SSE data:", e, jsonData);
            }
          }
        });
        await processStream();
      };
  
      processStream();
  
    } catch (error: unknown) {
      console.error("Error during initial request:", error);
      setIsProcessing(false);
      const msg = error instanceof Error ? error.message : String(error);
      setErrorMessage(msg);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">DCI Score Predictor</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-white p-6 rounded-lg shadow-lg">
        <input
          type="text"
          placeholder="Enter YouTube link"
          value={youtubeLink}
          onChange={(e) => setYoutubeLink(e.target.value)}
          className="w-full p-2 border rounded mb-4 text-gray-400"
          disabled={isProcessing}
        />
        <button
          type="submit"
          className="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600"
          disabled={isProcessing || !youtubeLink}
        >
          {isProcessing ? "Processing..." : "Predict Score"}
        </button>
      </form>

      {isProcessing && (
        <div className="mt-6 w-full max-w-md bg-gray-200 rounded-full overflow-hidden">
          <div
            className="bg-green-500 text-white text-center py-2 rounded-full"
            style={{ width: `${loadingProgress}%` }}
          >
            {loadingProgress}%
          </div>
        </div>
      )}

      {isProcessing && loadingStage && (
        <p className="mt-2 text-sm text-gray-600">{loadingStage}</p>
      )}

      {errorMessage && (
        <div className="mt-4 text-red-500 font-semibold">{errorMessage}</div>
      )}

      {prediction !== null && !isProcessing && !errorMessage && (
        <div className="mt-4 text-lg font-semibold text-gray-800">
          Predicted Score: {prediction}
        </div>
      )}
    </div>
  );
}
