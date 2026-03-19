import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

import { ActivityFeed } from '../src/components/ActivityFeed';
import { BayMaxFace } from '../src/components/BayMaxFace';
import { MemoryPanel } from '../src/components/MemoryPanel';
import type { ActivityEvent } from '../src/types/live';

// Mock the API module
vi.mock('../src/lib/api', () => ({
  getRecentMemories: vi.fn().mockResolvedValue({
    memories: [],
    facts: [],
    stats: { total_memories: 0, total_facts: 0, total_sessions: 0 },
    empty_reason: 'no active user',
  }),
  getUsers: vi.fn().mockResolvedValue([]),
  inspectDebugMemoryForUser: vi.fn().mockResolvedValue({
    memories: [],
    facts: [],
    stats: { total_memories: 0, total_facts: 0, total_sessions: 0 },
    empty_reason: 'no active user',
  }),
}));

describe('ActivityFeed', () => {
  it('renders empty state when no events', () => {
    render(<ActivityFeed events={[]} />);
    expect(screen.getByText('No events yet')).toBeInTheDocument();
  });

  it('renders events with correct color coding', () => {
    const events: ActivityEvent[] = [
      {
        id: '1',
        timestamp: new Date(),
        stage: 'recognition',
        status: 'complete',
        detail: 'User recognized',
        type: 'pipeline_event',
      },
      {
        id: '2',
        timestamp: new Date(),
        stage: 'memory',
        status: 'in_progress',
        detail: 'Retrieving memories',
        type: 'memory_event',
      },
      {
        id: '3',
        timestamp: new Date(),
        stage: 'safety',
        status: 'blocked',
        detail: 'Clinical language blocked',
        type: 'pipeline_event',
      },
    ];

    render(<ActivityFeed events={events} />);

    expect(screen.getByText('recognition')).toBeInTheDocument();
    expect(screen.getByText('memory')).toBeInTheDocument();
    expect(screen.getByText('safety')).toBeInTheDocument();
    expect(screen.getByText('3 events')).toBeInTheDocument();
  });

  it('displays event count', () => {
    const events: ActivityEvent[] = Array.from({ length: 5 }, (_, i) => ({
      id: String(i),
      timestamp: new Date(),
      stage: `stage-${i}`,
      status: 'complete',
      detail: `Detail ${i}`,
      type: 'pipeline_event',
    }));

    render(<ActivityFeed events={events} />);
    expect(screen.getByText('5 events')).toBeInTheDocument();
  });
});

describe('BayMaxFace', () => {
  it('renders offline state correctly', () => {
    render(<BayMaxFace state="OFFLINE" isSpeaking={false} />);
    expect(screen.getByText('OFFLINE')).toBeInTheDocument();
  });

  it('renders idle state correctly', () => {
    render(<BayMaxFace state="IDLE" isSpeaking={false} />);
    expect(screen.getByText('IDLE')).toBeInTheDocument();
  });

  it('renders listening state correctly', () => {
    render(<BayMaxFace state="LISTENING" isSpeaking={false} />);
    expect(screen.getByText('LISTENING')).toBeInTheDocument();
  });

  it('renders speaking state correctly', () => {
    render(<BayMaxFace state="SPEAKING" isSpeaking={true} />);
    expect(screen.getByText('SPEAKING')).toBeInTheDocument();
  });

  it('renders thinking state correctly', () => {
    render(<BayMaxFace state="THINKING" isSpeaking={false} />);
    expect(screen.getByText('THINKING')).toBeInTheDocument();
  });

  it('renders cooldown state correctly', () => {
    render(<BayMaxFace state="COOLDOWN" isSpeaking={false} />);
    expect(screen.getByText('COOLDOWN')).toBeInTheDocument();
  });
});

describe('MemoryPanel', () => {
  it('renders empty state when no memories', () => {
    render(
      <MemoryPanel
        memoryHits={[]}
        userId={null}
        refreshKey={0}
        recognitionState="no_face"
        sessionBinding="anonymous"
      />
    );
    expect(screen.getAllByText('no active user').length).toBeGreaterThan(0);
  });

  it('renders memory hits with scores', () => {
    const memoryHits = [
      { text: 'User likes coffee', score: 0.95 },
      { text: 'User works from home', score: 0.82 },
    ];

    render(
      <MemoryPanel
        memoryHits={memoryHits}
        userId="user-123"
        refreshKey={0}
        recognitionState="recognized_enrolled"
        sessionBinding="identified_user"
      />
    );

    expect(screen.getByText('User likes coffee')).toBeInTheDocument();
    expect(screen.getByText('User works from home')).toBeInTheDocument();
    expect(screen.getByText('score: 0.95')).toBeInTheDocument();
    expect(screen.getByText('score: 0.82')).toBeInTheDocument();
    expect(screen.getByText('2 hits')).toBeInTheDocument();
  });

  it('shows no user identified when userId is null', () => {
    render(
      <MemoryPanel
        memoryHits={[]}
        userId={null}
        refreshKey={0}
        recognitionState="unknown_user"
        sessionBinding="anonymous"
      />
    );
    expect(screen.getByText(/Memory unavailable until an enrolled user is recognized/i)).toBeInTheDocument();
  });
});
