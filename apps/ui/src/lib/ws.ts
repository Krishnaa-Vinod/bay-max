/**
 * WebSocket connection manager for live telemetry
 */

import type { WebSocketMessage, SnapshotMessage, PipelineEvent } from '@/types/live';

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketCallbacks {
  onSnapshot?: (data: SnapshotMessage) => void;
  onEvent?: (event: PipelineEvent) => void;
  onError?: (error: string) => void;
  onConnectionChange?: (state: ConnectionState) => void;
}

export class LiveWebSocket {
  private ws: WebSocket | null = null;
  private callbacks: WebSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private pingInterval: number | null = null;
  private url: string;

  constructor(url: string = '/ws/live') {
    // Use relative URL for WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    this.url = `${protocol}//${host}${url}`;
  }

  setCallbacks(callbacks: WebSocketCallbacks) {
    this.callbacks = callbacks;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.callbacks.onConnectionChange?.('connecting');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.callbacks.onConnectionChange?.('connected');
        this.startPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket closed');
        this.stopPing();
        this.callbacks.onConnectionChange?.('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.callbacks.onConnectionChange?.('error');
        this.callbacks.onError?.('WebSocket connection error');
      };
    } catch (e) {
      console.error('Failed to create WebSocket:', e);
      this.callbacks.onConnectionChange?.('error');
      this.scheduleReconnect();
    }
  }

  private handleMessage(message: WebSocketMessage) {
    switch (message.type) {
      case 'bootstrap':
      case 'snapshot':
        this.callbacks.onSnapshot?.(message as SnapshotMessage);
        break;
      case 'pipeline_event':
      case 'speech_event':
      case 'memory_event':
        this.callbacks.onEvent?.(message as PipelineEvent);
        break;
      case 'error':
        this.callbacks.onError?.(message.error);
        break;
    }
  }

  private startPing() {
    this.pingInterval = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, 30000);
  }

  private stopPing() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached');
      return;
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    setTimeout(() => this.connect(), delay);
  }

  disconnect() {
    this.stopPing();
    this.maxReconnectAttempts = 0; // Prevent reconnection
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
