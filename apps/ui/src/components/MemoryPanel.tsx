import { useState, useEffect } from 'react';
import type { MemoryHit } from '@/types/live';
import { getRecentMemories } from '@/lib/api';

interface MemoryPanelProps {
  memoryHits: MemoryHit[];
  userId: string | null;
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

export function MemoryPanel({ memoryHits, userId }: MemoryPanelProps) {
  const [facts, setFacts] = useState<Fact[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch memories when user changes
  useEffect(() => {
    if (!userId) {
      setFacts([]);
      setStats(null);
      return;
    }

    const fetchMemories = async () => {
      setLoading(true);
      try {
        const data = await getRecentMemories();
        setFacts(data.facts);
        setStats(data.stats);
      } catch (error) {
        console.error('Failed to fetch memories:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMemories();
    // Refresh every 10 seconds
    const interval = setInterval(fetchMemories, 10000);
    return () => clearInterval(interval);
  }, [userId]);

  return (
    <div className="flex-1 flex flex-col gap-4 overflow-hidden">
      {/* Retrieved memories */}
      <div className="flex-1 min-h-0 flex flex-col">
        <div className="text-xs text-gray-400 mb-2 flex justify-between">
          <span>Retrieved Memories</span>
          <span>{memoryHits.length} hits</span>
        </div>
        <div className="flex-1 overflow-y-auto bg-baymax-bg rounded p-2 space-y-2">
          {memoryHits.length === 0 ? (
            <div className="text-center text-gray-500 text-sm py-4">
              No memories retrieved
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
                    score: {hit.score.toFixed(2)}
                  </span>
                </div>
                <div className="text-gray-300">{hit.text}</div>
              </div>
            ))
          )}
        </div>
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
              {userId ? 'No facts stored' : 'No user identified'}
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
