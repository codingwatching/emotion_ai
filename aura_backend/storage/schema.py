"""Versioned schema surface for Aura's SQLite ledger."""

from __future__ import annotations

import sqlite3

from aura_backend.storage.models import StorageFailure

SCHEMA_VERSION = 1

_MIGRATION_1 = """
CREATE TABLE memory_scopes (
    scope_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES memory_scopes(scope_id),
    created_at TEXT NOT NULL,
    closed_at TEXT,
    UNIQUE(session_id, scope_id)
) STRICT;
CREATE INDEX sessions_scope_created ON sessions(scope_id, created_at, session_id);

CREATE TABLE turns (
    turn_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES memory_scopes(scope_id),
    session_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request_hash_version INTEGER NOT NULL CHECK(request_hash_version > 0),
    request_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    projection_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(projection_status IN ('pending', 'complete')),
    UNIQUE(scope_id, idempotency_key),
    UNIQUE(turn_id, scope_id),
    FOREIGN KEY(session_id, scope_id) REFERENCES sessions(session_id, scope_id)
) STRICT;
CREATE INDEX turns_scope_time ON turns(scope_id, occurred_at, turn_id);

CREATE TABLE events (
    event_pk INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    scope_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal IN (0, 1)),
    event_type TEXT NOT NULL DEFAULT 'message' CHECK(event_type = 'message'),
    actor TEXT NOT NULL CHECK(actor IN ('user', 'aura')),
    observed_at TEXT NOT NULL,
    content TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    UNIQUE(turn_id, ordinal),
    UNIQUE(event_id, scope_id),
    FOREIGN KEY(turn_id, scope_id) REFERENCES turns(turn_id, scope_id)
        ON DELETE CASCADE
) STRICT;
CREATE INDEX events_scope_time ON events(scope_id, observed_at, event_id);

CREATE TABLE derived_memories (
    memory_pk INTEGER PRIMARY KEY,
    memory_id TEXT NOT NULL UNIQUE,
    scope_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL
        CHECK(memory_kind IN ('fact', 'preference', 'episode', 'goal', 'relationship')),
    canonical_text TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    epistemic_status TEXT NOT NULL
        CHECK(epistemic_status IN ('observed', 'inferred', 'uncertain', 'disputed')),
    primary_source_event_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    UNIQUE(memory_id, scope_id),
    FOREIGN KEY(primary_source_event_id, scope_id)
        REFERENCES events(event_id, scope_id)
) STRICT;
CREATE INDEX memories_scope_time
    ON derived_memories(scope_id, created_at, memory_id);

CREATE TABLE memory_sources (
    memory_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'support'
        CHECK(relation IN ('support', 'contradict', 'context')),
    PRIMARY KEY(memory_id, event_id),
    FOREIGN KEY(memory_id, scope_id)
        REFERENCES derived_memories(memory_id, scope_id) ON DELETE CASCADE,
    FOREIGN KEY(event_id, scope_id)
        REFERENCES events(event_id, scope_id)
) STRICT;

CREATE TABLE memory_supersessions (
    old_memory_id TEXT NOT NULL,
    new_memory_id TEXT NOT NULL,
    basis_event_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(old_memory_id, new_memory_id),
    CHECK(old_memory_id <> new_memory_id),
    FOREIGN KEY(old_memory_id, scope_id)
        REFERENCES derived_memories(memory_id, scope_id),
    FOREIGN KEY(new_memory_id, scope_id)
        REFERENCES derived_memories(memory_id, scope_id),
    FOREIGN KEY(basis_event_id, scope_id)
        REFERENCES events(event_id, scope_id)
) STRICT;

CREATE TABLE memory_retractions (
    memory_id TEXT PRIMARY KEY,
    basis_event_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(memory_id, scope_id)
        REFERENCES derived_memories(memory_id, scope_id),
    FOREIGN KEY(basis_event_id, scope_id)
        REFERENCES events(event_id, scope_id)
) STRICT;

CREATE TABLE profile_versions (
    scope_id TEXT NOT NULL REFERENCES memory_scopes(scope_id),
    profile_version INTEGER NOT NULL CHECK(profile_version > 0),
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(scope_id, profile_version)
) STRICT;

CREATE TABLE legacy_sources (
    root_fingerprint TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    legacy_id TEXT NOT NULL,
    imported_origin_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(root_fingerprint, collection_name, legacy_id)
) STRICT;

CREATE TABLE projection_generations (
    generation_id TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    config_sha256 TEXT NOT NULL,
    metric TEXT NOT NULL CHECK(metric = 'cosine'),
    sqlite_watermark INTEGER NOT NULL CHECK(sqlite_watermark >= 0),
    build_status TEXT NOT NULL CHECK(build_status IN ('building', 'ready', 'failed')),
    created_at TEXT NOT NULL,
    completed_at TEXT
) STRICT;

CREATE TABLE retrieval_runs (
    run_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL REFERENCES memory_scopes(scope_id),
    query_sha256 TEXT NOT NULL,
    config_version INTEGER NOT NULL CHECK(config_version > 0),
    projection_generation_id TEXT REFERENCES projection_generations(generation_id),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE retrieval_candidates (
    run_id TEXT NOT NULL REFERENCES retrieval_runs(run_id) ON DELETE CASCADE,
    origin_id TEXT NOT NULL,
    origin_kind TEXT NOT NULL CHECK(origin_kind IN ('event', 'memory')),
    lexical_rank INTEGER,
    vector_rank INTEGER,
    neutral_score REAL NOT NULL CHECK(neutral_score >= 0.0 AND neutral_score <= 1.0),
    gate_status TEXT NOT NULL CHECK(gate_status IN ('eligible', 'rejected')),
    rejection_code TEXT,
    selected_rank INTEGER,
    PRIMARY KEY(run_id, origin_id)
) STRICT;

CREATE VIRTUAL TABLE event_fts USING fts5(
    content,
    content='events',
    content_rowid='event_pk',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER events_fts_insert AFTER INSERT ON events BEGIN
    INSERT INTO event_fts(rowid, content) VALUES (new.event_pk, new.content);
END;
CREATE TRIGGER events_fts_delete AFTER DELETE ON events BEGIN
    INSERT INTO event_fts(event_fts, rowid, content)
        VALUES ('delete', old.event_pk, old.content);
END;
CREATE TRIGGER events_fts_update AFTER UPDATE OF content ON events BEGIN
    INSERT INTO event_fts(event_fts, rowid, content)
        VALUES ('delete', old.event_pk, old.content);
    INSERT INTO event_fts(rowid, content) VALUES (new.event_pk, new.content);
END;

CREATE VIRTUAL TABLE memory_fts USING fts5(
    canonical_text,
    content='derived_memories',
    content_rowid='memory_pk',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER memories_fts_insert AFTER INSERT ON derived_memories BEGIN
    INSERT INTO memory_fts(rowid, canonical_text)
        VALUES (new.memory_pk, new.canonical_text);
END;
CREATE TRIGGER memories_fts_delete AFTER DELETE ON derived_memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, canonical_text)
        VALUES ('delete', old.memory_pk, old.canonical_text);
END;
CREATE TRIGGER memories_fts_update AFTER UPDATE OF canonical_text ON derived_memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, canonical_text)
        VALUES ('delete', old.memory_pk, old.canonical_text);
    INSERT INTO memory_fts(rowid, canonical_text)
        VALUES (new.memory_pk, new.canonical_text);
END;
"""


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply ordered forward migrations to ``connection``."""
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise StorageFailure("schema_too_new", identifier=str(current))
    if current == 0:
        try:
            connection.executescript(_MIGRATION_1)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except sqlite3.Error as error:
            raise StorageFailure("schema_migration_failed") from error


def rebuild_fts(connection: sqlite3.Connection) -> None:
    """Rebuild every external-content full-text projection."""
    try:
        connection.execute("INSERT INTO event_fts(event_fts) VALUES ('rebuild')")
        connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
    except sqlite3.Error as error:
        raise StorageFailure("fts_rebuild_failed") from error
