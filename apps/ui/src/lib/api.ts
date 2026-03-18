/**
 * HTTP API client for Bay-Max Live UI
 */

import type { UIBootstrapState, TextInputResponse, MicToggleResponse, MemoryData } from '@/types/live';

const API_BASE = '/v1/live';

export async function getUIState(): Promise<UIBootstrapState> {
  const response = await fetch(`${API_BASE}/ui-state`);
  if (!response.ok) {
    throw new Error(`Failed to fetch UI state: ${response.statusText}`);
  }
  return response.json();
}

export async function getLatestFrameUrl(): string {
  return `${API_BASE}/frame/latest?t=${Date.now()}`;
}

export async function submitTextInput(text: string): Promise<TextInputResponse> {
  const response = await fetch(`${API_BASE}/text-input`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(`Failed to submit text input: ${response.statusText}`);
  }
  return response.json();
}

export async function toggleMicrophone(action: 'start' | 'stop' | 'toggle'): Promise<MicToggleResponse> {
  const response = await fetch(`${API_BASE}/mic/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
  if (!response.ok) {
    throw new Error(`Failed to toggle microphone: ${response.statusText}`);
  }
  return response.json();
}

export async function getRecentMemories(): Promise<MemoryData> {
  const response = await fetch(`${API_BASE}/memory/recent`);
  if (!response.ok) {
    throw new Error(`Failed to fetch memories: ${response.statusText}`);
  }
  return response.json();
}
