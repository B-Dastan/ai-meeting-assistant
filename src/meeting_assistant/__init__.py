"""Meeting Assistant - AI-powered meeting notes generator."""

__version__ = "0.1.0"
__app_name__ = "Meeting Assistant"

from meeting_assistant.database import MeetingDatabase, Meeting

__all__ = [
    "__version__",
    "__app_name__",
    "MeetingDatabase",
    "Meeting",
]
