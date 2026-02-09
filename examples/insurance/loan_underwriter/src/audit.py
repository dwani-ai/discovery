import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

# SQLite audit database (file-based, zero config)
DB_PATH = Path("audit.db")

def init_audit_db():
    """Initialize SQLite audit table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            session_id TEXT,
            event_type TEXT,
            timestamp TEXT,
            payload TEXT,
            policy_version TEXT,
            PRIMARY KEY (session_id, timestamp)
        )
    """)
    conn.commit()
    conn.close()

def write_audit_event(session_id: str, event_type: str, payload: dict):
    """Append-only audit log."""
    init_audit_db()  # Ensure table exists
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.utcnow().isoformat()
    policy_version = payload.get("policy_version", "unknown")
    
    cursor.execute("""
        INSERT INTO audit_logs 
        (session_id, event_type, timestamp, payload, policy_version)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, event_type, timestamp, json.dumps(payload), policy_version))
    
    conn.commit()
    conn.close()
    print(f"📝 SQLite audit: {session_id} → {event_type}")

def enqueue_human_review(session_id: str, report: dict):
    """Simple file-based queue (or use Redis in prod)."""
    queue_file = Path("human_review_queue.jsonl")
    message = {
        "session_id": session_id,
        "report": report,
        "priority": "high" if report["risk_score"]["score"] < 50 else "medium",
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    with queue_file.open("a") as f:
        f.write(json.dumps(message) + "\n")
    
    print(f"👤 Human review queued: {session_id} (file: human_review_queue.jsonl)")

# Auto-init on import
init_audit_db()
