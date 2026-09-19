"""CivicProof — SQLite persistence layer."""
import json
import os
import sqlite3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "civicproof.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT 'other',
    category_label TEXT DEFAULT '',
    ai_confidence REAL DEFAULT 0,
    priority_score REAL DEFAULT 40,
    priority_label TEXT DEFAULT 'Medium',
    status TEXT DEFAULT 'open',
    flagged INTEGER DEFAULT 0,
    lat REAL, lng REAL,
    address TEXT DEFAULT '',
    reporter TEXT DEFAULT 'Anonymous Citizen',
    image_path TEXT DEFAULT '',
    after_image_path TEXT DEFAULT '',
    phash_a TEXT, phash_d TEXT,
    parent_id INTEGER,
    upvotes INTEGER DEFAULT 0,
    assigned_to TEXT DEFAULT '',
    ai_json TEXT DEFAULT '{}',
    pw_json TEXT DEFAULT '{}',
    resolved_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);
CREATE INDEX IF NOT EXISTS idx_reports_parent ON reports(parent_id);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);

CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER,
    ticket TEXT,
    event TEXT NOT NULL,
    detail TEXT DEFAULT '',
    actor TEXT DEFAULT 'system',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_report ON activity(report_id);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)


def row_to_dict(row):
    d = dict(row)
    for k in ("ai_json", "pw_json"):
        try:
            d[k.replace("_json", "")] = json.loads(d.get(k) or "{}")
        except Exception:
            d[k.replace("_json", "")] = {}
    d.pop("ai_json", None)
    d.pop("pw_json", None)
    return d


def get_report(report_id):
    with connect() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        return row_to_dict(row) if row else None


def list_reports(where="", params=(), order="priority_score DESC"):
    q = f"SELECT * FROM reports {where} ORDER BY {order}, id DESC"
    with connect() as conn:
        return [row_to_dict(r) for r in conn.execute(q, params).fetchall()]


def ticket_for(report_id):
    return f"CP-{time.strftime('%y')}-{report_id:04d}"


def insert_report(fields):
    fields = dict(fields)
    now = time.time()
    fields.setdefault("created_at", now)
    fields.setdefault("updated_at", now)
    fields["ticket"] = f"TMP-{int(now * 1e6)}"  # placeholder, replaced with CP-YY-#### below
    with connect() as conn:
        cols = ", ".join(fields.keys())
        ph = ", ".join("?" for _ in fields)
        cur = conn.execute(f"INSERT INTO reports ({cols}) VALUES ({ph})", tuple(fields.values()))
        rid = cur.lastrowid
        ticket = ticket_for(rid)
        conn.execute("UPDATE reports SET ticket=? WHERE id=?", (ticket, rid))
        return rid, ticket


def update_report(report_id, fields):
    fields = dict(fields)
    fields["updated_at"] = time.time()
    sets = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE reports SET {sets} WHERE id=?", (*fields.values(), report_id))


def log_event(report_id, event, detail="", actor="system", ticket=None):
    with connect() as conn:
        conn.execute(
            "INSERT INTO activity (report_id, ticket, event, detail, actor, created_at) VALUES (?,?,?,?,?,?)",
            (report_id, ticket, event, detail, actor, time.time()),
        )


def get_activity(limit=40, report_id=None):
    with connect() as conn:
        if report_id:
            rows = conn.execute(
                "SELECT * FROM activity WHERE report_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
                (report_id, limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM activity ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
