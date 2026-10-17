"""PostgreSQL access layer with automatic reconnect + retry."""
import logging
import time

import numpy as np
import psycopg2

log = logging.getLogger("facetracker")

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id     SERIAL PRIMARY KEY,
    embedding   BYTEA NOT NULL,
    first_seen  TIMESTAMPTZ NOT NULL,
    image_path  TEXT
);
ALTER TABLE faces ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'face';
ALTER TABLE faces ADD COLUMN IF NOT EXISTS body_embedding BYTEA;
CREATE TABLE IF NOT EXISTS events (
    id          SERIAL PRIMARY KEY,
    face_id     INTEGER NOT NULL REFERENCES faces(face_id),
    event_type  TEXT NOT NULL CHECK (event_type IN ('entry','exit')),
    ts          TIMESTAMPTZ NOT NULL,
    image_path  TEXT,
    track_id    INTEGER
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          SERIAL PRIMARY KEY,
    face_id     INTEGER NOT NULL REFERENCES faces(face_id),
    kind        TEXT NOT NULL CHECK (kind IN ('face','body')),
    vec         BYTEA NOT NULL
);
CREATE TABLE IF NOT EXISTS track_spans (
    id          SERIAL PRIMARY KEY,
    face_id     INTEGER NOT NULL REFERENCES faces(face_id),
    run_id      TEXT NOT NULL,
    first_frame INTEGER NOT NULL,
    last_frame  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
"""


class Database:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.conn = None
        self._connect()
        self._run(lambda cur: cur.execute(SCHEMA))

    def _connect(self):
        c = self.cfg
        self.conn = psycopg2.connect(host=c["host"], port=c["port"], dbname=c["name"],
                                     user=c["user"], password=c["password"])

    def _run(self, fn, retries: int = 3):
        """Run fn(cursor) in its own transaction; reconnect and retry on failure."""
        for attempt in range(1, retries + 1):
            try:
                with self.conn:                      # commit / rollback
                    with self.conn.cursor() as cur:
                        return fn(cur)
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                log.error("DB error (attempt %d/%d): %s", attempt, retries, e)
                time.sleep(1)
                try:
                    self._connect()
                except Exception as ce:  # noqa: BLE001
                    log.error("DB reconnect failed: %s", ce)
        raise RuntimeError("database unavailable")

    # ---- faces -----------------------------------------------------------
    def add_face(self, embedding, ts, image_path: str, kind: str = "face", body_emb=None) -> int:
        """Register a visitor. kind='face' stores the ArcFace embedding; kind='body' (face never seen) stores none."""
        raw = embedding.astype(np.float32).tobytes() if embedding is not None else b""
        braw = body_emb.astype(np.float32).tobytes() if body_emb is not None else None
        def fn(cur):
            cur.execute("INSERT INTO faces (embedding, first_seen, image_path, kind, body_embedding) "
                        "VALUES (%s,%s,%s,%s,%s) RETURNING face_id", (raw, ts, image_path, kind, braw))
            return cur.fetchone()[0]
        return self._run(fn)

    def load_faces(self):
        def fn(cur):
            cur.execute("SELECT face_id, embedding FROM faces WHERE kind = 'face' ORDER BY face_id")
            return [(i, np.frombuffer(bytes(b), dtype=np.float32)) for i, b in cur.fetchall()]
        return self._run(fn)

    def load_bodies(self):
        """(id, kind, body_embedding|None) for every registered visitor."""
        def fn(cur):
            cur.execute("SELECT face_id, kind, body_embedding FROM faces ORDER BY face_id")
            return [(i, k, np.frombuffer(bytes(b), dtype=np.float32) if b else None) for i, k, b in cur.fetchall()]
        return self._run(fn)

    def set_body_embedding(self, face_id: int, emb: np.ndarray):
        self._run(lambda cur: cur.execute("UPDATE faces SET body_embedding=%s WHERE face_id=%s",
                                          (emb.astype(np.float32).tobytes(), face_id)))

    def upgrade_to_face(self, face_id: int, emb: np.ndarray):
        """A body-only visitor whose face was later seen becomes a face visitor."""
        self._run(lambda cur: cur.execute("UPDATE faces SET kind='face', embedding=%s WHERE face_id=%s",
                                          (emb.astype(np.float32).tobytes(), face_id)))

    def add_embedding(self, face_id: int, kind: str, vec: np.ndarray):
        """Persist one gallery embedding (kind = 'face' | 'body') so identities survive restarts and new videos."""
        self._run(lambda cur: cur.execute("INSERT INTO embeddings (face_id, kind, vec) VALUES (%s,%s,%s)",
                                          (face_id, kind, vec.astype(np.float32).tobytes())))

    def load_embeddings(self):
        def fn(cur):
            cur.execute("SELECT face_id, kind, vec FROM embeddings ORDER BY id")
            return [(i, k, np.frombuffer(bytes(v), dtype=np.float32)) for i, k, v in cur.fetchall()]
        return self._run(fn)

    def add_span(self, face_id: int, run_id: str, first_frame: int, last_frame: int):
        """When (in frames of one run) this visitor was on screen: two visitors on screen together are different people."""
        self._run(lambda cur: cur.execute(
            "INSERT INTO track_spans (face_id, run_id, first_frame, last_frame) VALUES (%s,%s,%s,%s)",
            (face_id, run_id, first_frame, last_frame)))

    def load_spans(self):
        def fn(cur):
            cur.execute("SELECT face_id, run_id, first_frame, last_frame FROM track_spans")
            return cur.fetchall()
        return self._run(fn)

    def merge_visitors(self, keep: int, drop: int):
        """Fold visitor `drop` into `keep`: events, embeddings and spans move over; the duplicate row is deleted."""
        def fn(cur):
            for table in ("events", "embeddings", "track_spans"):
                cur.execute(f"UPDATE {table} SET face_id=%s WHERE face_id=%s", (keep, drop))
            cur.execute("SELECT kind, embedding FROM faces WHERE face_id=%s", (drop,))
            dkind, demb = cur.fetchone()
            cur.execute("SELECT kind FROM faces WHERE face_id=%s", (keep,))
            if dkind == "face" and cur.fetchone()[0] == "body":     # the merged visitor now has a face identity
                cur.execute("UPDATE faces SET kind='face', embedding=%s WHERE face_id=%s", (demb, keep))
            cur.execute("DELETE FROM faces WHERE face_id=%s", (drop,))
        self._run(fn)

    def count_unique(self) -> int:
        def fn(cur):
            cur.execute("SELECT COUNT(*) FROM faces")
            return cur.fetchone()[0]
        return self._run(fn)

    def count_by_kind(self) -> dict:
        def fn(cur):
            cur.execute("SELECT kind, COUNT(*) FROM faces GROUP BY kind")
            return dict(cur.fetchall())
        return self._run(fn)

    # ---- events ----------------------------------------------------------
    def add_event(self, face_id: int, event_type: str, ts, image_path: str, track_id: int):
        self._run(lambda cur: cur.execute(
            "INSERT INTO events (face_id,event_type,ts,image_path,track_id) VALUES (%s,%s,%s,%s,%s)",
            (face_id, event_type, ts, image_path, track_id)))

    def close(self):
        if self.conn:
            self.conn.close()
