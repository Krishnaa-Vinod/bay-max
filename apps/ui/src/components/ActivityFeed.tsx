import { useEffect, useRef } from 'react';
import type { ActivityEvent } from '@/types/live';
import clsx from 'clsx';

interface ActivityFeedProps {
  events: ActivityEvent[];
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'complete':
      return 'text-green-400 border-l-green-500';
    case 'in_progress':
      return 'text-yellow-400 border-l-yellow-500';
    case 'pending':
      return 'text-blue-400 border-l-blue-500';
    case 'blocked':
    case 'suppressed':
      return 'text-orange-400 border-l-orange-500';
    case 'error':
      return 'text-red-400 border-l-red-500';
    default:
      return 'text-gray-400 border-l-gray-500';
  }
}

function getTypeIcon(type: string): string {
  switch (type) {
    case 'pipeline_event':
      return '⚙️';
    case 'speech_event':
      return '🎤';
    case 'memory_event':
      return '💾';
    case 'error':
      return '❌';
    default:
      return '📋';
  }
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function ActivityFeed({ events }: ActivityFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);

  // Auto-scroll to bottom unless user is scrolling up
  useEffect(() => {
    const container = containerRef.current;
    if (!container || isUserScrolling.current) return;

    container.scrollTop = container.scrollHeight;
  }, [events]);

  // Detect user scroll
  const handleScroll = () => {
    const container = containerRef.current;
    if (!container) return;

    const isAtBottom = container.scrollHeight - container.scrollTop <= container.clientHeight + 50;
    isUserScrolling.current = !isAtBottom;
  };

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="text-xs text-gray-400 mb-2 flex justify-between">
        <span>Activity Feed</span>
        <span>{events.length} events</span>
      </div>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto bg-baymax-bg rounded p-2 space-y-1 activity-feed"
      >
        {events.length === 0 ? (
          <div className="text-center text-gray-500 text-sm py-4">
            No events yet
          </div>
        ) : (
          events.map((event) => (
            <div
              key={event.id}
              className={clsx(
                'text-xs px-2 py-1 border-l-2 bg-gray-800/50 rounded-r',
                getStatusColor(event.status)
              )}
            >
              <div className="flex items-center gap-2">
                <span className="opacity-60">{getTypeIcon(event.type)}</span>
                <span className="text-gray-500">{formatTime(event.timestamp)}</span>
                <span className="font-medium">{event.stage}</span>
                <span className="text-gray-500">•</span>
                <span className={clsx('capitalize', getStatusColor(event.status))}>
                  {event.status}
                </span>
              </div>
              {event.detail && (
                <div className="text-gray-400 mt-0.5 pl-6 truncate">
                  {event.detail}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
