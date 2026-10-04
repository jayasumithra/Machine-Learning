"""
=============================================================================
AI-Based Tuberculosis (TB) Screening from Chest X-Ray
SQLite Database Module
=============================================================================
Provides clean data persistence for prediction history, timestamps,
probabilities, and Grad-CAM artifact references.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def get_connection() -> sqlite3.Connection:
    """Creates a connection to the SQLite database with dictionary-like row access."""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the SQLite database tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            tb_probability REAL NOT NULL,
            normal_probability REAL NOT NULL,
            gradcam_image TEXT,
            execution_time_ms REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(
    image_filename: str,
    model_name: str,
    prediction: str,
    tb_probability: float,
    normal_probability: float,
    gradcam_image: Optional[str] = None,
    execution_time_ms: float = 0.0
) -> int:
    """
    Saves a single screening prediction record into the database.
    Returns the newly inserted record ID.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            date, time, image_filename, model_name,
            prediction, tb_probability, normal_probability,
            gradcam_image, execution_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        date_str,
        time_str,
        image_filename,
        model_name,
        prediction,
        round(float(tb_probability), 2),
        round(float(normal_probability), 2),
        gradcam_image or "",
        round(float(execution_time_ms), 1)
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves prediction history records in descending chronological order."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, date, time, image_filename, model_name,
               prediction, tb_probability, normal_probability,
               gradcam_image, execution_time_ms
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    history = [dict(row) for row in rows]
    conn.close()
    return history

def delete_prediction(pred_id: int) -> bool:
    """Deletes a specific prediction entry by its ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_history() -> int:
    """Deletes all prediction records from the database. Returns count of deleted records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count

def get_history_stats() -> Dict[str, Any]:
    """Returns aggregated summary statistics of all recorded predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'TB'")
    tb_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Normal'")
    normal_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(tb_probability) FROM predictions WHERE prediction = 'TB'")
    avg_tb_conf = cursor.fetchone()[0] or 0.0

    conn.close()
    return {
        "total_screenings": total,
        "tb_screenings": tb_count,
        "normal_screenings": normal_count,
        "avg_tb_confidence": round(avg_tb_conf, 2)
    }

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", config.DATABASE_PATH)
