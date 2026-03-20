import { useState, useEffect } from 'react';
import type { MemoryHit } from '@/types/live';
import { getRecentMemories, getUsers, inspectDebugMemoryForUser } from '@/lib/api';

interface MemoryPanelProps {
  memoryHits: MemoryHit[];
  userId: string | null;
  refreshKey: number;
  recognitionState: 'no_face' | 'face_seen_unknown' | 'known_low_confidence' | 'known_attached';
  sessionBinding: 'anonymous' | 'identified_user';
}

interface Fact {
  key: string;
  value: string;
  confidence: number;
}

interface MemoryStats {
  total_memories: number;
  total_facts: number;
  total_sessions: number;
}

export function MemoryPanel({ memoryHits, userId, refreshKey, recognitionState, sessionBinding }: MemoryPanelProps) {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [emptyReason, setEmptyReason] = useState('no active user');
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [recognitionReason, setRecognitionReason] = useState('no face detected');
  const [memoryUnavailableHint, setMemoryUnavailableHint] = useState('Memory unavailable until an enrolled user is recognized.');
  const [debugUsers, setDebugUsers] = useState<Array<{ id: string; display_name: string }>>([]);
  const [debugUserId, setDebugUserId] = useState<string>('');
  const [debugMode, setDebugMode] = useState(false);

  // Fetch memories when user changes
  useEffect(() => {
    if (!userId && !debugMode) {
      setFacts([]);
      setStats(null);
      setEmptyReason('no active user');
      setMemoryError(null);
      return;
    }

    const fetchMemories = async () => {
      setLoading(true);
      try {
        const data = debugMode && debugUserId
          ? await inspectDebugMemoryForUser(debugUserId)
          : await getRecentMemories();
        setFacts(data.facts);
        setStats(data.stats);
        setEmptyReason(data.empty_reason || '');
        setRecognitionReason(data.recognition_reason || '');
        setMemoryUnavailableHint(data.memory_unavailable_hint || '');
        setMemoryError(data.error ?? null);
      } catch (error) {
        setMemoryError(error instanceof Error ? error.message : 'Failed to fetch memories');
        setEmptyReason('memory service error');
      } finally {
        setLoading(false);
      }
    };

    fetchMemories();
    const interval = setInterval(fetchMemories, 5000);
    return () => clearInterval(interval);
  }, [userId, refreshKey, debugMode, debugUserId]);

  useEffect(() => {
    const loadUsers = async () => {
      try {
        const users = await getUsers();
        setDebugUsers(users);
        if (!debugUserId && users.length > 0) {
          setDebugUserId(users[0].id);
        }
      } catch {
        // Optional debug control only.
      }
    };
    loadUsers();
  }, [debugUserId]);

  const retrievedEmptyReason = memoryError
    ? 'memory service error'
    : (!userId ? 'no active user' : 'no retrieved memories yet');

  return (
    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
      {/* Retrieved memories */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="text-xs text-gray-400 mb-2 flex justify-between">
          <span>Retrieved Memories</span>
          <span>{memoryHits.length} hits</span>
        </div>
        <div className="text-xs text-gray-500 mb-2">
          {recognitionReason || (
            recognitionState === 'known_attached'
              ? 'known user attached'
              : recognitionState === 'known_low_confidence'
                ? 'known user, low confidence'
                : recognitionState === 'face_seen_unknown'
                  ? 'face detected, unknown user'
                  : 'no face detected'
          )}
          {sessionBinding === 'anonymous' ? ' | binding: anonymous' : ' | binding: identified_user'}
        </div>
        {sessionBinding === 'anonymous' && (
          <div className="text-xs text-amber-300 mb-2">
            {memoryUnavailableHint || 'Memory unavailable until an enrolled user is recognized.'}
          </div>
        )}
        <div className="flex-1 overflow-y-auto bg-baymax-bg rounded p-2 space-y-2">
          {memoryHits.length === 0 ? (
            <div className="text-center text-gray-500 text-sm py-4">
              {retrievedEmptyReason}
            </div>
          ) : (
            memoryHits.map((hit, idx) => (
              <div
                key={`${idx}-${hit.text.slice(0, 20)}`}
                className="text-sm p-2 bg-gray-800/50 rounded border-l-2 border-blue-500"
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="text-xs text-blue-400 font-medium">
                    #{idx + 1}
                  </span>
                  <span className="text-xs text-gray-500">
                    {hit.score !== null ? `score: ${hit.score.toFixed(2)}` : ''}
                  </span>
                </div>
                <div className="text-gray-300">{hit.text}</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Debug-only memory inspector */}
      <div className="bg-baymax-bg rounded p-3 border border-amber-700/40">
        <div className="text-xs text-amber-300 mb-2">DEBUG ONLY: Memory Inspector</div>
        <div className="flex items-center gap-2 mb-2">
          <label className="text-xs text-gray-300 flex items-center gap-1">
            <input
              type="checkbox"
              checked={debugMode}
              onChange={(e) => setDebugMode(e.target.checked)}
            />
            Inspect selected enrolled user
          </label>
        </div>
        <select
          className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-white"
          value={debugUserId}
          onChange={(e) => setDebugUserId(e.target.value)}
          disabled={!debugMode || debugUsers.length === 0}
        >
          {debugUsers.length === 0 && <option value="">No enrolled users found</option>}
          {debugUsers.map((u) => (
            <option key={u.id} value={u.id}>{u.display_name} ({u.id.slice(0, 8)})</option>
          ))}
        </select>
      </div>

      {/* Semantic facts */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="text-xs text-gray-400 mb-2 flex justify-between">
          <span>Semantic Facts</span>
          <span>{facts.length} facts</span>
        </div>
        <div className="flex-1 overflow-y-auto bg-baymax-bg rounded p-2 space-y-2">
          {loading ? (
            <div className="text-center text-gray-500 text-sm py-4">
              Loading...
            </div>
          ) : facts.length === 0 ? (
            <div className="text-center text-gray-500 text-sm py-4">
              {emptyReason || (userId ? 'no retrieved memories yet' : 'no active user')}
            </div>
          ) : (
            facts.map((fact, idx) => (
              <div
                key={`${idx}-${fact.key}`}
                className="text-sm p-2 bg-gray-800/50 rounded"
              >
                <div className="flex justify-between items-start">
                  <span className="text-purple-400 font-medium">{fact.key}</span>
                  <span className="text-xs text-gray-500">
                    {(fact.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="text-gray-300 mt-1">{fact.value}</div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Memory stats */}
      <div className="bg-baymax-bg rounded p-3">
        <div className="text-xs text-gray-400 mb-2">Memory Stats</div>
        {memoryError && (
          <div className="text-xs text-red-400 mb-2">
            {memoryError}
          </div>
        )}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-lg font-bold text-white">
              {stats?.total_memories ?? 0}
            </div>
            <div className="text-xs text-gray-500">Memories</div>
          </div>
          <div>
            <div className="text-lg font-bold text-white">
              {stats?.total_facts ?? 0}
            </div>
            <div className="text-xs text-gray-500">Facts</div>
          </div>
          <div>
            <div className="text-lg font-bold text-white">
              {stats?.total_sessions ?? 0}
            </div>
            <div className="text-xs text-gray-500">Sessions</div>
          </div>
        </div>
      </div>
    </div>
  );
}
