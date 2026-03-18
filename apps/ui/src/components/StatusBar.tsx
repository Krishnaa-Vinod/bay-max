import type { ConnectionState } from '@/lib/ws';
import type { PipelineState } from '@/types/live';
import clsx from 'clsx';

interface StatusBarProps {
  connectionState: ConnectionState;
  pipelineState: PipelineState;
  uptime: number;
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);

  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

export function StatusBar({ connectionState, pipelineState, uptime }: StatusBarProps) {
  return (
    <div className="flex items-center gap-4">
      {/* Connection status */}
      <div className="flex items-center gap-2">
        <div
          className={clsx('w-2 h-2 rounded-full', {
            'bg-green-500': connectionState === 'connected',
            'bg-yellow-500 animate-pulse': connectionState === 'connecting',
            'bg-red-500': connectionState === 'disconnected' || connectionState === 'error',
          })}
        />
        <span className="text-xs text-gray-400 capitalize">{connectionState}</span>
      </div>

      {/* Pipeline state */}
      <span
        className={clsx('status-badge', {
          'status-offline': pipelineState === 'OFFLINE',
          'status-idle': pipelineState === 'IDLE',
          'status-active': pipelineState === 'ACTIVE',
          'status-listening': pipelineState === 'LISTENING',
          'status-thinking': pipelineState === 'THINKING',
          'status-speaking': pipelineState === 'SPEAKING',
          'status-cooldown': pipelineState === 'COOLDOWN',
        })}
      >
        {pipelineState}
      </span>

      {/* Uptime */}
      <div className="text-xs text-gray-500">
        Uptime: {formatUptime(uptime)}
      </div>
    </div>
  );
}
