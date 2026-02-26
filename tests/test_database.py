"""Tests for the MeetingDatabase module."""

import tempfile
from pathlib import Path

import pytest  # type: ignore[import-untyped]

from meeting_assistant.database import Meeting, MeetingDatabase


@pytest.fixture
def db(tmp_path: Path) -> MeetingDatabase:
    """Create a temporary database for testing."""
    return MeetingDatabase(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def sample_meeting() -> Meeting:
    """Create a sample meeting for testing."""
    return Meeting(
        title="Sprint Planning",
        transcript="We discussed the upcoming sprint goals and assigned tasks.",
        summary="The team planned the next sprint and assigned responsibilities.",
        key_points=["New feature prioritized", "Bug fixes scheduled"],
        action_items=["Alice to design UI mockups", "Bob to fix login bug"],
    )


class TestMeeting:
    """Tests for the Meeting dataclass."""

    def test_default_values(self):
        meeting = Meeting()
        assert meeting.id is None
        assert meeting.title == "Untitled Meeting"
        assert meeting.transcript == ""
        assert meeting.key_points == []
        assert meeting.action_items == []

    def test_to_dict(self, sample_meeting: Meeting):
        data = sample_meeting.to_dict()
        assert data["title"] == "Sprint Planning"
        assert len(data["key_points"]) == 2
        assert len(data["action_items"]) == 2

    def test_validation(self):
        """Test that Pydantic rejects invalid types."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            Meeting(title=123)  # type: ignore[arg-type]

    def test_model_dump(self, sample_meeting: Meeting):
        """Test Pydantic's model_dump serialization."""
        data = sample_meeting.model_dump()
        assert isinstance(data, dict)
        assert data["title"] == "Sprint Planning"
        assert isinstance(data["key_points"], list)


class TestMeetingDatabase:
    """Tests for the MeetingDatabase class."""

    def test_save_and_retrieve(self, db: MeetingDatabase, sample_meeting: Meeting):
        meeting_id = db.save_meeting(sample_meeting)
        assert meeting_id is not None

        retrieved = db.get_meeting(meeting_id)
        assert retrieved is not None
        assert retrieved.title == "Sprint Planning"
        assert retrieved.key_points == [
            "New feature prioritized", "Bug fixes scheduled"]

    def test_update_meeting(self, db: MeetingDatabase, sample_meeting: Meeting):
        db.save_meeting(sample_meeting)
        sample_meeting.title = "Updated Sprint Planning"
        db.save_meeting(sample_meeting)

        assert sample_meeting.id is not None
        retrieved = db.get_meeting(sample_meeting.id)
        assert retrieved is not None
        assert retrieved.title == "Updated Sprint Planning"

    def test_get_all_meetings(self, db: MeetingDatabase):
        db.save_meeting(Meeting(title="Meeting 1"))
        db.save_meeting(Meeting(title="Meeting 2"))
        db.save_meeting(Meeting(title="Meeting 3"))

        meetings = db.get_all_meetings()
        assert len(meetings) == 3

    def test_delete_meeting(self, db: MeetingDatabase, sample_meeting: Meeting):
        db.save_meeting(sample_meeting)
        assert sample_meeting.id is not None
        result = db.delete_meeting(sample_meeting.id)
        assert result is not None  # Returns audio_path (empty string)
        assert db.get_meeting(sample_meeting.id) is None

    def test_delete_nonexistent(self, db: MeetingDatabase):
        assert db.delete_meeting(999) is None

    def test_search_meetings(self, db: MeetingDatabase):
        db.save_meeting(Meeting(title="Python Workshop",
                        transcript="We learned about decorators"))
        db.save_meeting(Meeting(title="Sprint Planning",
                        transcript="Assigned tasks"))

        results = db.search_meetings("Python")
        assert len(results) == 1
        assert results[0].title == "Python Workshop"

        results = db.search_meetings("tasks")
        assert len(results) == 1

    def test_get_nonexistent_meeting(self, db: MeetingDatabase):
        assert db.get_meeting(999) is None
