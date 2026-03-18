import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { LiveWebSocket, ConnectionState } from '../src/lib/ws';

describe('LiveWebSocket', () => {
  let mockWebSocket: any;

  beforeEach(() => {
    mockWebSocket = {
      readyState: WebSocket.CONNECTING,
      send: vi.fn(),
      close: vi.fn(),
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
    };

    vi.stubGlobal('WebSocket', vi.fn(() => mockWebSocket));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('creates WebSocket with correct URL', () => {
    // Mock window.location
    vi.stubGlobal('location', {
      protocol: 'http:',
      host: 'localhost:3000',
    });

    const ws = new LiveWebSocket('/ws/live');
    ws.connect();

    expect(WebSocket).toHaveBeenCalledWith('ws://localhost:3000/ws/live');
  });

  it('calls onConnectionChange when connected', () => {
    const onConnectionChange = vi.fn();
    const ws = new LiveWebSocket();
    ws.setCallbacks({ onConnectionChange });
    ws.connect();

    // Simulate open event
    mockWebSocket.readyState = WebSocket.OPEN;
    mockWebSocket.onopen?.();

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

    mockWebSocket.readyState = WebSocket.OPEN;
    expect(ws.isConnected).toBe(true);

    mockWebSocket.readyState = WebSocket.CLOSED;
    expect(ws.isConnected).toBe(false);
  });
});
