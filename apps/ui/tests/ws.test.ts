import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Create mock before importing the module
const mockWebSocket = {
  readyState: 0, // CONNECTING
  send: vi.fn(),
  close: vi.fn(),
  onopen: null as (() => void) | null,
  onclose: null as (() => void) | null,
  onmessage: null as ((event: { data: string }) => void) | null,
  onerror: null as ((error: Event) => void) | null,
};

const WebSocketMock = vi.fn(() => mockWebSocket);

// Mock window.location before importing
Object.defineProperty(globalThis, 'window', {
  value: {
    location: {
      protocol: 'http:',
      host: 'localhost:3000',
    },
    setInterval: globalThis.setInterval,
    clearInterval: globalThis.clearInterval,
  },
  writable: true,
});

// Mock WebSocket global before importing
(globalThis as any).WebSocket = WebSocketMock;
(globalThis as any).WebSocket.CONNECTING = 0;
(globalThis as any).WebSocket.OPEN = 1;
(globalThis as any).WebSocket.CLOSING = 2;
(globalThis as any).WebSocket.CLOSED = 3;

// Now import the module
import { LiveWebSocket } from '../src/lib/ws';

describe('LiveWebSocket', () => {
  beforeEach(() => {
    // Reset the mock for each test
    WebSocketMock.mockClear();
    mockWebSocket.send.mockClear();
    mockWebSocket.close.mockClear();
    mockWebSocket.readyState = 0; // CONNECTING
    mockWebSocket.onopen = null;
    mockWebSocket.onclose = null;
    mockWebSocket.onmessage = null;
    mockWebSocket.onerror = null;
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  it('creates WebSocket with correct URL', () => {
    const ws = new LiveWebSocket('/ws/live');
    ws.connect();

    expect(WebSocketMock).toHaveBeenCalledWith('ws://localhost:3000/ws/live');
  });

  it('calls onConnectionChange only after first snapshot', () => {
    const onConnectionChange = vi.fn();
    const ws = new LiveWebSocket();
    ws.setCallbacks({ onConnectionChange });
    ws.connect();

    // Should be called with 'connecting' first
    expect(onConnectionChange).toHaveBeenCalledWith('connecting');

    // Simulate open event
    mockWebSocket.readyState = 1; // OPEN
    mockWebSocket.onopen?.();

    // Still connecting until telemetry bootstrap/snapshot arrives
    expect(onConnectionChange).not.toHaveBeenCalledWith('connected');

    const bootstrapMessage = {
      type: 'bootstrap',
      timestamp: '2026-03-18T12:00:00Z',
      session: { id: 'test', state: 'active', duration_sec: 100, turn_count: 5 },
      perception: {},
      pipeline_state: 'ACTIVE',
    };

    mockWebSocket.onmessage?.({ data: JSON.stringify(bootstrapMessage) });

    expect(onConnectionChange).toHaveBeenCalledWith('connected');
  });

  it('calls onSnapshot for snapshot messages', () => {
    const onSnapshot = vi.fn();
    const ws = new LiveWebSocket();
    ws.setCallbacks({ onSnapshot });
    ws.connect();

    const snapshotMessage = {
      type: 'snapshot',
      timestamp: '2026-03-18T12:00:00Z',
      session: { id: 'test', state: 'active', duration_sec: 100, turn_count: 5 },
      perception: {},
      pipeline_state: 'ACTIVE',
    };

    mockWebSocket.onmessage?.({ data: JSON.stringify(snapshotMessage) });

    expect(onSnapshot).toHaveBeenCalledWith(snapshotMessage);
  });

  it('calls onEvent for pipeline events', () => {
    const onEvent = vi.fn();
    const ws = new LiveWebSocket();
    ws.setCallbacks({ onEvent });
    ws.connect();

    const eventMessage = {
      type: 'pipeline_event',
      timestamp: '2026-03-18T12:00:00Z',
      stage: 'recognition',
      status: 'complete',
      detail: 'User recognized',
    };

    mockWebSocket.onmessage?.({ data: JSON.stringify(eventMessage) });

    expect(onEvent).toHaveBeenCalledWith(eventMessage);
  });

  it('calls onError for error messages', () => {
    const onError = vi.fn();
    const ws = new LiveWebSocket();
    ws.setCallbacks({ onError });
    ws.connect();

    const errorMessage = {
      type: 'error',
      error: 'Something went wrong',
    };

    mockWebSocket.onmessage?.({ data: JSON.stringify(errorMessage) });

    expect(onError).toHaveBeenCalledWith('Something went wrong');
  });

  it('isConnected returns true when WebSocket is open', () => {
    const ws = new LiveWebSocket();
    ws.connect();

    mockWebSocket.readyState = 1; // OPEN
    expect(ws.isConnected).toBe(true);

    // Note: Changing readyState on the mock doesn't affect the ws.ws instance
    // because the instance was already created. We need to test a fresh instance
    // or disconnect and reconnect.
    ws.disconnect();
    expect(ws.isConnected).toBe(false);
  });
});
