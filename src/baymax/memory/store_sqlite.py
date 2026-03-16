"""SQLite-backed metadata store implementation."""

import json
import sqlite3
from datetime import datetime
from uuid import UUID

from baymax.core.enums import MemoryStatus, MemoryType
from baymax.memory.interfaces import MetadataStore
from baymax.schemas.memory import (
    EpisodicMemory,
    MemoryQuery,
    MemoryQueryResult,
    SemanticFact,
)
from baymax.schemas.session import Session
from baymax.schemas.user import FaceEnrollment, UserProfile

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS face_enrollments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    enrolled_at TEXT NOT NULL,
    encoding_ref TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    status TEXT DEFAULT 'active',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    frame_count INTEGER DEFAULT 0,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    content TEXT NOT NULL,
    salience REAL DEFAULT 0.5,
    emotion_context TEXT,
    evidence_refs TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    memory_type TEXT DEFAULT 'episodic'
);

CREATE TABLE IF NOT EXISTS semantic_facts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    first_observed TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    evidence_refs TEXT DEFAULT '[]',
    status TEXT DEFAULT 'active',
    memory_type TEXT DEFAULT 'semantic'
);
"""


class SQLiteMetadataStore(MetadataStore):
    """SQLite-backed implementation of MetadataStore."""

    def __init__(self, db_path: str = "baymax.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Create the database and tables."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    async def create_user(self, user: UserProfile) -> UserProfile:
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO users"
            " (id, display_name, notes, created_at, updated_at, is_active)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(user.id),
                user.display_name,
                user.notes,
                user.created_at.isoformat(),
                user.updated_at.isoformat(),
                int(user.is_active),
            ),
        )
        conn.commit()
        return user

    async def get_user(self, user_id: UUID) -> UserProfile | None:
        conn = self._ensure_conn()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (str(user_id),)).fetchone()
        if row is None:
            return None
        return UserProfile(
            id=UUID(row["id"]),
            display_name=row["display_name"],
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            is_active=bool(row["is_active"]),
        )

    async def create_face_enrollment(self, enrollment: FaceEnrollment) -> FaceEnrollment:
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO face_enrollments"
            " (id, user_id, enrolled_at, encoding_ref, confidence, status)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(enrollment.id),
                str(enrollment.user_id),
                enrollment.enrolled_at.isoformat(),
                enrollment.encoding_ref,
                enrollment.confidence,
                enrollment.status,
            ),
        )
        conn.commit()
        return enrollment

    async def create_session(self, session: Session) -> Session:
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO sessions"
            " (id, user_id, status, started_at, ended_at,"
            " frame_count, notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(session.id),
                str(session.user_id) if session.user_id else None,
                session.status.value,
                session.started_at.isoformat(),
                session.ended_at.isoformat()
                if session.ended_at
                else None,
                session.frame_count,
                session.notes,
            ),
        )
        conn.commit()
        return session

    async def get_session(self, session_id: UUID) -> Session | None:
        conn = self._ensure_conn()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (str(session_id),)).fetchone()
        if row is None:
            return None
        return Session(
            id=UUID(row["id"]),
            user_id=UUID(row["user_id"]) if row["user_id"] else None,
            status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            frame_count=row["frame_count"],
            notes=row["notes"],
        )

    async def store_episodic_memory(self, memory: EpisodicMemory) -> EpisodicMemory:
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO episodic_memories"
            " (id, user_id, session_id, timestamp, content,"
            " salience, emotion_context, evidence_refs,"
            " status, memory_type)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(memory.id),
                str(memory.user_id),
                str(memory.session_id),
                memory.timestamp.isoformat(),
                memory.content,
                memory.salience,
                memory.emotion_context,
                json.dumps(memory.evidence_refs),
                memory.status.value,
                memory.memory_type.value,
            ),
        )
        conn.commit()
        return memory

    async def store_semantic_fact(self, fact: SemanticFact) -> SemanticFact:
        conn = self._ensure_conn()
        conn.execute(
            "INSERT INTO semantic_facts"
            " (id, user_id, content, confidence,"
            " first_observed, last_confirmed,"
            " evidence_refs, status, memory_type)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(fact.id),
                str(fact.user_id),
                fact.content,
                fact.confidence,
                fact.first_observed.isoformat(),
                fact.last_confirmed.isoformat(),
                json.dumps(fact.evidence_refs),
                fact.status.value,
                fact.memory_type.value,
            ),
        )
        conn.commit()
        return fact

    async def query_memories(self, query: MemoryQuery) -> MemoryQueryResult:
        conn = self._ensure_conn()
        episodic = []
        semantic = []

        if MemoryType.EPISODIC in query.memory_types:
            rows = conn.execute(
                "SELECT * FROM episodic_memories"
                " WHERE user_id = ? AND status = ?"
                " ORDER BY salience DESC LIMIT ?",
                (str(query.user_id), MemoryStatus.ACTIVE.value, query.limit),
            ).fetchall()
            for row in rows:
                episodic.append(EpisodicMemory(
                    id=UUID(row["id"]),
                    user_id=UUID(row["user_id"]),
                    session_id=UUID(row["session_id"]),
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    content=row["content"],
                    salience=row["salience"],
                    emotion_context=row["emotion_context"],
                    evidence_refs=json.loads(row["evidence_refs"]),
                    status=row["status"],
                    memory_type=row["memory_type"],
                ))

        if MemoryType.SEMANTIC in query.memory_types:
            rows = conn.execute(
                "SELECT * FROM semantic_facts"
                " WHERE user_id = ? AND status = ?"
                " ORDER BY confidence DESC LIMIT ?",
                (str(query.user_id), MemoryStatus.ACTIVE.value, query.limit),
            ).fetchall()
            for row in rows:
                semantic.append(SemanticFact(
                    id=UUID(row["id"]),
                    user_id=UUID(row["user_id"]),
                    content=row["content"],
                    confidence=row["confidence"],
                    first_observed=datetime.fromisoformat(row["first_observed"]),
                    last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                    evidence_refs=json.loads(row["evidence_refs"]),
                    status=row["status"],
                    memory_type=row["memory_type"],
                ))

        total = len(episodic) + len(semantic)
        return MemoryQueryResult(
            user_id=query.user_id,
            query=query.query,
            episodic_memories=episodic,
            semantic_facts=semantic,
            total_count=total,
        )

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
