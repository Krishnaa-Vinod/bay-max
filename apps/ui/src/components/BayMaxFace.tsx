import { useState, useEffect } from 'react';
import type { PipelineState } from '@/types/live';
import clsx from 'clsx';

interface BayMaxFaceProps {
  state: PipelineState;
  isSpeaking: boolean;
}

export function BayMaxFace({ state, isSpeaking }: BayMaxFaceProps) {
  const [blinkFrame, setBlinkFrame] = useState(false);
  const [mouthOpen, setMouthOpen] = useState(false);

  // Blink animation
  useEffect(() => {
    if (state === 'OFFLINE') return;

    const blinkInterval = setInterval(() => {
      setBlinkFrame(true);
      setTimeout(() => setBlinkFrame(false), 150);
    }, 3500 + Math.random() * 1000); // 3.5-4.5 seconds

    return () => clearInterval(blinkInterval);
  }, [state]);

  // Speaking mouth animation
  useEffect(() => {
    if (!isSpeaking) {
      setMouthOpen(false);
      return;
    }

    const mouthInterval = setInterval(() => {
      setMouthOpen((prev) => !prev);
    }, 150);

    return () => clearInterval(mouthInterval);
  }, [isSpeaking]);

  // Eye position based on state
  const getEyeTransform = () => {
    switch (state) {
      case 'THINKING':
        return 'translateY(-3px)'; // Look up
      case 'LISTENING':
        return 'scale(1.1)'; // Wider eyes
      default:
        return 'none';
    }
  };

  // Eye scale during blink
  const eyeScaleY = blinkFrame ? 0.1 : 1;

  return (
    <div className="flex flex-col items-center">
      {/* Face container */}
      <div
        className={clsx(
          'relative w-40 h-40 rounded-full flex items-center justify-center transition-all duration-300',
          state === 'OFFLINE' ? 'bg-gray-800' : 'bg-white baymax-face-glow'
        )}
      >
        {/* Eyes container */}
        <div className="flex gap-6" style={{ transform: getEyeTransform() }}>
          {/* Left eye */}
          <div
            className={clsx(
              'w-8 h-8 rounded-full transition-all duration-150',
              state === 'OFFLINE' ? 'bg-gray-600' : 'bg-gray-900'
            )}
            style={{ transform: `scaleY(${eyeScaleY})` }}
          />
          {/* Right eye */}
          <div
            className={clsx(
              'w-8 h-8 rounded-full transition-all duration-150',
              state === 'OFFLINE' ? 'bg-gray-600' : 'bg-gray-900'
            )}
            style={{ transform: `scaleY(${eyeScaleY})` }}
          />
        </div>

        {/* Connecting line between eyes */}
        <div
          className={clsx(
            'absolute w-6 h-1 rounded-full',
            state === 'OFFLINE' ? 'bg-gray-600' : 'bg-gray-900'
          )}
          style={{ transform: getEyeTransform() }}
        />

        {/* Speaking indicator mouth */}
        {isSpeaking && (
          <div
            className="absolute bottom-10 w-4 rounded-full bg-gray-900 transition-all duration-100"
            style={{ height: mouthOpen ? '8px' : '2px' }}
          />
        )}
      </div>

      {/* State label */}
      <div className="mt-3">
        <span
          className={clsx('status-badge', {
            'status-offline': state === 'OFFLINE',
            'status-idle': state === 'IDLE',
            'status-active': state === 'ACTIVE',
            'status-listening': state === 'LISTENING',
            'status-thinking': state === 'THINKING',
            'status-speaking': state === 'SPEAKING',
            'status-cooldown': state === 'COOLDOWN',
          })}
        >
          {state}
        </span>
      </div>
    </div>
  );
}
