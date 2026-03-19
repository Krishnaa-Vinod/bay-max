import type { SessionState } from '@/types/live';
import clsx from 'clsx';

interface SessionInfoProps {
  sessionId: string | null;
  sessionState: SessionState;
  sessionBinding: 'anonymous' | 'identified_user';
  duration: number;
  turnCount: number;
  lastEvent: string;
}

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function SessionInfo({
  sessionId,
  sessionState,
  sessionBinding,
  duration,
  turnCount,
  lastEvent,
}: SessionInfoProps) {
  return (
    <div className="bg-baymax-bg rounded p-3">
      <div className="text-xs text-gray-400 mb-2">Session Info</div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">ID:</span>
          <span className="text-white font-mono text-xs">
            {sessionId ? sessionId.slice(0, 8) : 'none'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">State:</span>
          <span
            className={clsx('capitalize', {
              'text-green-400': sessionState === 'active',
              'text-yellow-400': sessionState === 'paused',
              'text-gray-400': sessionState === 'idle',
              'text-red-400': sessionState === 'ended',
            })}
          >
            {sessionState}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Duration:</span>
          <span className="text-white">{formatDuration(duration)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Turns:</span>
          <span className="text-white">{turnCount}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Binding:</span>
          <span className={clsx('text-xs', {
            'text-cyan-300': sessionBinding === 'identified_user',
            'text-gray-400': sessionBinding === 'anonymous',
          })}>
            {sessionBinding === 'identified_user' ? 'user-linked' : 'anonymous'}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Last Event:</span>
          <span className="text-cyan-400 text-xs truncate max-w-[120px]">
            {lastEvent || 'none'}
          </span>
        </div>
      </div>
    </div>
  );
}
