"""Database module for storing and retrieving meeting records."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class Meeting(BaseModel):
    """Represents a single meeting record."""

    id: Optional[int] = None
    title: str = "Untitled Meeting"
    date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M"))
    transcript: str = ""
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    audio_path: str = ""

    # Allow arbitrary types (needed for sqlite3.Row compatibility)
    model_config = {"from_attributes": True}

    def to_dict(self) -> dict:
        """Convert meeting to dictionary."""
        return self.model_dump()


class MeetingDatabase:
    """SQLite database for meeting storage."""

    def __init__(self, db_path: str = "meetings.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL DEFAULT 'Untitled Meeting',
                    date TEXT NOT NULL,
                    transcript TEXT DEFAULT '',
                    summary TEXT DEFAULT '',
                    key_points TEXT DEFAULT '[]',
                    action_items TEXT DEFAULT '[]',
                    audio_path TEXT DEFAULT ''
                )
            """)
            conn.commit()

    def save_meeting(self, meeting: Meeting) -> int:
        """Save a meeting to the database. Returns the meeting ID."""
        with sqlite3.connect(self.db_path) as conn:
            if meeting.id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO meetings (title, date, transcript, summary, key_points, action_items, audio_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meeting.title,
                        meeting.date,
                        meeting.transcript,
                        meeting.summary,
                        json.dumps(meeting.key_points),
                        json.dumps(meeting.action_items),
                        meeting.audio_path,
                    ),
                )
                meeting.id = cursor.lastrowid or 0
            else:
                conn.execute(
                    """
                    UPDATE meetings
                    SET title=?, date=?, transcript=?, summary=?, key_points=?, action_items=?, audio_path=?
                    WHERE id=?
                    """,
                    (
                        meeting.title,
                        meeting.date,
                        meeting.transcript,
                        meeting.summary,
                        json.dumps(meeting.key_points),
                        json.dumps(meeting.action_items),
                        meeting.audio_path,
                        meeting.id,
                    ),
                )
            conn.commit()
        return meeting.id  # type: ignore[return-value]

    def get_meeting(self, meeting_id: int) -> Optional[Meeting]:
        """Retrieve a single meeting by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_meeting(row)

    def get_all_meetings(self) -> list[Meeting]:
        """Retrieve all meetings, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM meetings ORDER BY date DESC"
            ).fetchall()
            return [self._row_to_meeting(row) for row in rows]

    def delete_meeting(self, meeting_id: int) -> Optional[str]:
        """Delete a meeting by ID. Returns the audio_path if deleted, None otherwise."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT audio_path FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()
            if row is None:
                return None
            audio_path = row["audio_path"]
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
            conn.commit()
            return audio_path

    def search_meetings(self, query: str) -> list[Meeting]:
        """Search meetings by title, transcript, or summary."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM meetings
                WHERE title LIKE ? OR transcript LIKE ? OR summary LIKE ?
                ORDER BY date DESC
                """,
                (f"%{query}%", f"%{query}%", f"%{query}%"),
            ).fetchall()
            return [self._row_to_meeting(row) for row in rows]

    @staticmethod
    def _row_to_meeting(row: sqlite3.Row) -> Meeting:
        """Convert a database row to a Meeting object."""
        return Meeting(
            id=row["id"],
            title=row["title"],
            date=row["date"],
            transcript=row["transcript"],
            summary=row["summary"],
            key_points=json.loads(row["key_points"]),
            action_items=json.loads(row["action_items"]),
            audio_path=row["audio_path"],
        )
