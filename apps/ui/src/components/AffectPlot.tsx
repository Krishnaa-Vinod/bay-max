interface AffectPlotProps {
  history: Array<{
    timestamp: number;
    valence: number;
    arousal: number;
  }>;
  currentValence: number;
  currentArousal: number;
  confidence: number;
  backend: string;
}

export function AffectPlot({
  history,
  currentValence,
  currentArousal,
  confidence,
  backend,
}: AffectPlotProps) {
  // SVG dimensions
  const width = 200;
  const height = 80;
  const padding = 10;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  // Scale values to plot coordinates
  const scaleX = (i: number) => padding + (i / Math.max(history.length - 1, 1)) * plotWidth;
  const scaleY = (v: number) => padding + ((1 - v) / 2 + 0.5) * plotHeight; // Map -1..1 to plotHeight..0

  // Generate path for valence line
  const valencePath = history.length > 1
    ? history.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(p.valence)}`).join(' ')
    : '';

  // Generate path for arousal line
  const arousalPath = history.length > 1
    ? history.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i)} ${scaleY(p.arousal * 2 - 1)}`).join(' ')
    : '';

  // Get valence color
  const getValenceColor = (v: number) => {
    if (v > 0.2) return '#22c55e'; // Green - positive
    if (v < -0.2) return '#3b82f6'; // Blue - subdued
    return '#9ca3af'; // Gray - neutral
  };

  return (
    <div className="bg-baymax-bg rounded p-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-gray-400">Affect</span>
        <span className="text-xs text-gray-500">
          {backend || 'unavailable'}
        </span>
      </div>

      {confidence < 0.3 ? (
        <div className="text-center text-gray-500 text-sm py-4">
          Low confidence / No face detected
        </div>
      ) : (
        <>
          {/* Mini plot */}
          <svg width={width} height={height} className="w-full">
            {/* Background */}
            <rect
              x={padding}
              y={padding}
              width={plotWidth}
              height={plotHeight}
              fill="#1a1f25"
              rx={4}
            />

            {/* Zero line */}
            <line
              x1={padding}
              y1={height / 2}
              x2={width - padding}
              y2={height / 2}
              stroke="#374151"
              strokeDasharray="4 2"
            />

            {/* Valence line */}
            {valencePath && (
              <path
                d={valencePath}
                fill="none"
                stroke={getValenceColor(currentValence)}
                strokeWidth={2}
              />
            )}

            {/* Arousal line */}
            {arousalPath && (
              <path
                d={arousalPath}
                fill="none"
                stroke="#eab308"
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            )}

            {/* Current valence point */}
            {history.length > 0 && (
              <circle
                cx={scaleX(history.length - 1)}
                cy={scaleY(currentValence)}
                r={4}
                fill={getValenceColor(currentValence)}
              />
            )}
          </svg>

          {/* Values */}
          <div className="flex justify-between text-xs mt-2">
            <div>
              <span className="text-gray-400">Valence:</span>
              <span
                className="ml-1 font-medium"
                style={{ color: getValenceColor(currentValence) }}
              >
                {currentValence > 0 ? '+' : ''}{currentValence.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Arousal:</span>
              <span className="ml-1 font-medium text-yellow-500">
                {currentArousal.toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Conf:</span>
              <span className="ml-1 text-gray-300">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Legend */}
          <div className="flex gap-4 text-xs text-gray-500 mt-1">
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-green-500" />
              <span>Valence</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-0.5 bg-yellow-500" style={{ borderStyle: 'dashed' }} />
              <span>Arousal</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
