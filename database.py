import sqlite3
from configurations import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            speed REAL,
            plate TEXT,
            timestamp TEXT,
            frame_number INTEGER,
            status TEXT,
            violation_type TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_violation(track_id, speed, plate, timestamp, frame_number, status, violation_type, expired_set):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO violations (track_id, speed, plate, timestamp, frame_number, status, violation_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (track_id, speed, plate, timestamp, frame_number, status, violation_type))

    if plate in expired_set:
        cursor.execute("""
            INSERT INTO violations (track_id, speed, plate, timestamp, frame_number, status, violation_type)
            VALUES (?, ?, ?, ?, ?, ?, "Expired Insurance")
        """, (track_id, speed, plate, timestamp, frame_number, status))

    conn.commit()
    conn.close()
