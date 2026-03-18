import { useState } from 'react';
import { submitTextInput, toggleMicrophone } from '@/lib/api';
import clsx from 'clsx';

interface TextInputProps {
  speechEnabled: boolean;
  isListening: boolean;
}

export function TextInput({ speechEnabled, isListening }: TextInputProps) {
  const [text, setText] = useState('');
  const [sending, setSending] = useState(false);
  const [micLoading, setMicLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || sending) return;

    setSending(true);
    try {
      const response = await submitTextInput(text.trim());
      if (response.success) {
        setText('');
      } else {
        console.error('Text input failed:', response.error);
      }
    } catch (error) {
      console.error('Failed to submit text:', error);
    } finally {
      setSending(false);
    }
  };

  const handleMicToggle = async () => {
    if (micLoading || !speechEnabled) return;

    setMicLoading(true);
    try {
      await toggleMicrophone('toggle');
    } catch (error) {
      console.error('Failed to toggle mic:', error);
    } finally {
      setMicLoading(false);
    }
  };

  return (
    <div className="mt-auto">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a message..."
          disabled={sending}
          className="flex-1 bg-baymax-bg border border-baymax-border rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-baymax-primary disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={sending || !text.trim()}
          className="px-4 py-2 bg-baymax-primary text-white rounded text-sm font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {sending ? 'Sending...' : 'Send'}
        </button>
        <button
          type="button"
          onClick={handleMicToggle}
          disabled={micLoading || !speechEnabled}
          className={clsx(
            'p-2 rounded transition-colors',
            isListening
              ? 'bg-red-600 hover:bg-red-700 text-white'
              : 'bg-baymax-bg border border-baymax-border text-gray-400 hover:text-white hover:border-gray-500',
            (micLoading || !speechEnabled) && 'opacity-50 cursor-not-allowed'
          )}
          title={speechEnabled ? (isListening ? 'Stop listening' : 'Start listening') : 'Speech input disabled'}
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
        </button>
      </form>
      {!speechEnabled && (
        <div className="text-xs text-gray-500 mt-1">
          Speech input is disabled. Enable it in backend settings.
        </div>
      )}
    </div>
  );
}
