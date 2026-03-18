import { useState, useEffect } from 'react';

interface CameraFeedProps {
  frameUrl: string;
  isConnected: boolean;
}

export function CameraFeed({ frameUrl, isConnected }: CameraFeedProps) {
  const [imageError, setImageError] = useState(false);
  const [imageSrc, setImageSrc] = useState(frameUrl);

  // Refresh image periodically
  useEffect(() => {
    if (!isConnected) return;

    const interval = setInterval(() => {
      setImageSrc(`/v1/live/frame/latest?t=${Date.now()}`);
      setImageError(false);
    }, 200); // 5 FPS refresh

    return () => clearInterval(interval);
  }, [isConnected]);

  return (
    <div className="relative bg-black rounded overflow-hidden aspect-video">
      {!isConnected || imageError ? (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-center">
            <svg
              className="w-16 h-16 mx-auto text-gray-600 mb-2"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            <p className="text-gray-500 text-sm">
              {!isConnected ? 'Connecting...' : 'No video feed'}
            </p>
          </div>
        </div>
      ) : (
        <img
          src={imageSrc}
          alt="Live camera feed"
          className="w-full h-full object-contain"
          onError={() => setImageError(true)}
        />
      )}

      {/* Connection indicator */}
      <div className="absolute top-2 right-2">
        <div
          className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
            }`}
        />
      </div>
    </div>
  );
}
