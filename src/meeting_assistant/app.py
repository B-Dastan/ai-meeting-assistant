"""Meeting Assistant - Streamlit UI Application."""

import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from meeting_assistant import __app_name__, __version__
from meeting_assistant.database import Meeting, MeetingDatabase
from meeting_assistant.exporter import MeetingExporter
from meeting_assistant.summarizer import Summarizer
from meeting_assistant.transcriber import Transcriber

# --- Configuration ---
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000


def _check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and available."""
    return shutil.which("ffmpeg") is not None


# --- Initialize Services (cached) ---
@st.cache_resource
def get_transcriber() -> Transcriber:
    return Transcriber()


@st.cache_resource
def get_summarizer() -> Summarizer:
    return Summarizer()


@st.cache_resource
def get_database() -> MeetingDatabase:
    return MeetingDatabase()


@st.cache_resource
def get_exporter() -> MeetingExporter:
    return MeetingExporter()


# --- Session State Initialization ---
def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "transcript": "",
        "current_meeting": None,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Main App ---
def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=__app_name__,
        page_icon="🎙️",
        layout="wide",
    )

    init_session_state()
    db = get_database()

    # --- Header ---
    st.title(__app_name__)
    st.caption(f"v{__version__} — AI-powered meeting notes generator")

    # --- System Checks ---
    if not Path(".env").exists():
        st.error(
            "**`.env` file not found!** Copy the example and configure it:\n\n"
            "```\ncp .env.example .env\n```"
        )
        st.stop()

    if not _check_ffmpeg():
        st.error(
            "**FFmpeg not found!** Whisper requires FFmpeg to process audio.\n\n"
            "Install it:\n"
            "- **Windows:** `winget install ffmpeg`\n"
            "- **Mac:** `brew install ffmpeg`\n"
            "- **Linux:** `sudo apt install ffmpeg`\n\n"
            "Then restart this app."
        )
        st.stop()

    # --- Sidebar: Meeting History ---
    with st.sidebar:
        st.header("Meeting History")
        meetings = db.get_all_meetings()

        if meetings:
            for meeting in meetings:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(
                        meeting.title,
                        key=f"meeting_{meeting.id}",
                        use_container_width=True,
                    ):
                        st.session_state.current_meeting = meeting
                with col2:
                    if st.button("🗑️", key=f"delete_{meeting.id}"):
                        if meeting.id is not None:
                            audio_path = db.delete_meeting(meeting.id)
                            if audio_path and Path(audio_path).exists():
                                Path(audio_path).unlink()
                        if (
                            st.session_state.current_meeting
                            and st.session_state.current_meeting.id == meeting.id
                        ):
                            st.session_state.current_meeting = None
                        st.rerun()
        else:
            st.info("No meetings yet. Record or upload one!")

    # --- Main Content ---
    tab_record, tab_upload, tab_view = st.tabs(
        ["Record", "Upload Audio", "View Meeting"]
    )

    # --- Tab 1: Record ---
    with tab_record:
        st.subheader("Record a Meeting")
        st.write("Click the microphone to start recording, click again to stop.")

        try:
            from audio_recorder_streamlit import audio_recorder

            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#6c757d",
                icon_size="2x",
                pause_threshold=300.0,  # 5 minutes max
            )

            if audio_bytes and len(audio_bytes) > 1000:
                st.audio(audio_bytes, format="audio/wav")

                if st.button("Process Recording", type="primary"):
                    # Save audio bytes to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_path = UPLOADS_DIR / f"recording_{timestamp}.wav"
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    _process_audio(str(audio_path))
            elif audio_bytes:
                st.warning("Recording is too short. Please try again.")

        except ImportError:
            st.error(
                "Recording requires the `audio_recorder_streamlit` package.\n\n"
                "Install it with: `pip install audio-recorder-streamlit`"
            )
            st.info(
                "You can still use the **Upload Audio** tab to process audio files.")

    # --- Tab 2: Upload ---
    with tab_upload:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Upload a recorded meeting audio file",
        )

        if uploaded_file is not None:
            # Save uploaded file
            audio_path = UPLOADS_DIR / uploaded_file.name
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
            st.audio(uploaded_file)

            if st.button("Process Audio", type="primary"):
                _process_audio(str(audio_path))

    # --- Tab 3: View Meeting ---
    with tab_view:
        meeting = st.session_state.current_meeting
        if meeting is None:
            st.info("Select a meeting from the sidebar, or record/upload one.")
        else:
            _display_meeting(meeting)


def _process_audio(audio_path: str) -> None:
    """Transcribe and summarize an audio file."""
    transcriber = get_transcriber()
    summarizer = get_summarizer()
    db = get_database()

    with st.status("Processing meeting...", expanded=True) as status:
        # Step 1: Check audio duration
        import soundfile as sf
        try:
            audio_info = sf.info(audio_path)
            if audio_info.duration < 1.0:
                status.update(label="Audio too short", state="error")
                st.warning(
                    "Recording is too short (less than 1 second). Please record a longer clip.")
                return
        except Exception:
            pass  # Let Whisper handle any file issues

        # Step 2: Transcribe
        st.write("Transcribing audio with Whisper...")
        result = transcriber.transcribe_file(audio_path)
        transcript = result["text"]

        if not transcript.strip():
            status.update(label="No speech detected", state="error")
            st.warning(
                "Could not detect any speech in the audio. Please try again.")
            return

        st.write(f"Transcribed ({len(transcript.split())} words)")

        # Step 2: Summarize
        st.write("Generating summary with AI...")
        ai_result = summarizer.process_meeting(transcript)
        st.write("Summary generated")

        # Step 3: Save
        st.write("Saving meeting...")
        meeting = Meeting(
            title=ai_result["title"],
            transcript=transcript,
            summary=ai_result["summary"],
            key_points=ai_result["key_points"],
            action_items=ai_result["action_items"],
            audio_path=audio_path,
        )
        db.save_meeting(meeting)
        st.session_state.current_meeting = meeting
        status.update(label="Meeting processed!", state="complete")

    st.rerun()


def _display_meeting(meeting: Meeting) -> None:
    """Display meeting details."""
    exporter = get_exporter()

    st.header(meeting.title)
    st.caption(meeting.date)

    # Export button
    if st.button("Export to PDF"):
        pdf_path = exporter.export_to_pdf(meeting)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                data=f.read(),
                file_name=pdf_path.name,
                mime="application/pdf",
            )

    # Summary
    st.subheader("Summary")
    st.write(meeting.summary)

    # Key Points
    if meeting.key_points:
        st.subheader("Key Points")
        for point in meeting.key_points:
            st.markdown(f"- {point}")

    # Action Items
    if meeting.action_items:
        st.subheader("Action Items")
        for i, item in enumerate(meeting.action_items):
            st.checkbox(item, key=f"action_{i}")

    # Q&A
    st.subheader("Ask About This Meeting")
    question = st.text_input("Ask a question about the meeting...")
    if question:
        summarizer = get_summarizer()
        with st.spinner("Thinking..."):
            answer = summarizer.answer_question(meeting.transcript, question)
        st.write(answer)

    # Full Transcript (expandable)
    with st.expander("Full Transcript"):
        st.text(meeting.transcript)


# Allow running with: streamlit run app.py
if __name__ == "__main__":
    main()
