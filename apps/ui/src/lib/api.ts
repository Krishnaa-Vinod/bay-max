/**
 * HTTP API client for Bay-Max Live UI
 */

import type { UIBootstrapState, TextInputResponse, MicToggleResponse, MemoryData } from '@/types/live';

const API_BASE = '/v1/live';

async function extractErrorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data?.detail === 'string' && data.detail) {
      return data.detail;
    }
    if (typeof data?.error === 'string' && data.error) {
      return data.error;
    }
  } catch {
    // ignore json parse errors and use fallback
  }
  return fallback;
}

export async function getUIState(): Promise<UIBootstrapState> {
  const response = await fetch(`${API_BASE}/ui-state`);
  if (!response.ok) {
    const message = await extractErrorMessage(response, `Failed to fetch UI state: ${response.statusText}`);
    throw new Error(message);
  }
  return response.json();
}

export function getLatestFrameUrl(): string {
  return `${API_BASE}/frame/latest?t=${Date.now()}`;
}

export async function submitTextInput(text: string): Promise<TextInputResponse> {
  const response = await fetch(`${API_BASE}/text-input`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    const message = await extractErrorMessage(response, `Failed to submit text input: ${response.statusText}`);
    throw new Error(message);
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
    const message = await extractErrorMessage(response, `Failed to toggle microphone: ${response.statusText}`);
    throw new Error(message);
  }
  return response.json();
}

export async function getRecentMemories(): Promise<MemoryData> {
  const response = await fetch(`${API_BASE}/memory/recent`);
  if (!response.ok) {
    const message = await extractErrorMessage(response, `Failed to fetch memories: ${response.statusText}`);
    throw new Error(message);
  }
  return response.json();
}

export async function inspectDebugMemoryForUser(userId: string): Promise<MemoryData> {
  const response = await fetch(`${API_BASE}/memory/debug/inspect/${encodeURIComponent(userId)}`);
  if (!response.ok) {
    const message = await extractErrorMessage(response, `Failed to inspect debug memory: ${response.statusText}`);
    throw new Error(message);
  }
  return response.json();
}

export async function getUsers(): Promise<Array<{ id: string; display_name: string }>> {
  const response = await fetch('/v1/users');
  if (!response.ok) {
    const message = await extractErrorMessage(response, `Failed to fetch users: ${response.statusText}`);
    throw new Error(message);
  }
  const data = await response.json();
  if (!Array.isArray(data)) {
    return [];
  }
  return data.map((item: any) => ({
    id: String(item.id),
    display_name: String(item.display_name ?? 'unknown'),
  }));
}
