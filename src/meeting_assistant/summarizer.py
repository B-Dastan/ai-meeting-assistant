"""AI-powered meeting summarization using local LLM via Docker Model Runner."""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class Summarizer:
    """Generates meeting summaries, key points, and action items using an LLM."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "not-needed",
    ):
        """
        Initialize the summarizer.

        Args:
            base_url: LLM API endpoint (defaults to LLM_BASE_URL env var).
            model: Model name to use (defaults to LLM_MODEL env var).
            api_key: API key (not needed for local models).
        """
        self.base_url = base_url or os.environ["LLM_BASE_URL"]
        self.model = model or os.environ["LLM_MODEL"]
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    def _chat(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request to the LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return (response.choices[0].message.content or "").strip()

    def _chunk_transcript(self, text: str, max_words: int = 800) -> list[str]:
        """Split long transcripts into manageable chunks."""
        words = text.split()
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    def _parse_json_list(self, response: str) -> list[str]:
        """Robustly parse a JSON array from LLM response."""
        cleaned = response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return [
                line.strip().lstrip("•-*0123456789. \"")
                for line in response.split("\n")
                if line.strip()
                and line.strip() not in ["[", "]"]
                and not line.strip().startswith(
                    ("import", "def ", "for ", "action_",
                     "meeting_", "#", "json", "print")
                )
            ]

    def generate_summary(self, transcript: str) -> str:
        """
        Generate a professional summary of the meeting transcript.

        Args:
            transcript: The full meeting transcript text.

        Returns:
            A concise professional summary.
        """
        system_prompt = (
            "You are a professional meeting notes assistant. "
            "Generate a clear, concise summary of the meeting transcript. "
            "Focus on decisions made, topics discussed, and overall outcomes. "
            "Write in a professional tone using past tense."
        )
        chunks = self._chunk_transcript(transcript)
        if len(chunks) == 1:
            return self._chat(system_prompt, f"Meeting transcript:\n\n{transcript}")

        partial = [self._chat(
            system_prompt, f"Meeting transcript:\n\n{c}") for c in chunks]
        combined = "\n\n".join(partial)
        return self._chat(system_prompt, f"Meeting transcript:\n\n{combined}")

    def extract_key_points(self, transcript: str) -> list[str]:
        """
        Extract key discussion points from the transcript.

        Args:
            transcript: The full meeting transcript text.

        Returns:
            List of key points as strings.
        """
        transcript = " ".join(transcript.split()[:800])  # truncate for safety
        system_prompt = (
            "OUTPUT FORMAT: JSON array only. Example: [\"Point 1\", \"Point 2\"]\n"
            "RULES: No code. No explanation. No markdown. No imports. No variables.\n"
            "TASK: Extract the most important key points from the meeting transcript below."
        )
        response = self._chat(
            system_prompt, f"Meeting transcript:\n\n{transcript}")
        return self._parse_json_list(response)

    def extract_action_items(self, transcript: str) -> list[str]:
        """
        Extract action items and tasks from the transcript.

        Args:
            transcript: The full meeting transcript text.

        Returns:
            List of action items as strings.
        """
        transcript = " ".join(transcript.split()[:800])  # truncate for safety
        system_prompt = (
            "OUTPUT FORMAT: JSON array only. Example: [\"Action 1\", \"Action 2\"]\n"
            "RULES: No code. No explanation. No markdown. No imports. No variables.\n"
            "TASK: Extract all action items and follow-ups from the meeting transcript below."
        )
        response = self._chat(
            system_prompt, f"Meeting transcript:\n\n{transcript}")
        return self._parse_json_list(response)

    def generate_title(self, transcript: str) -> str:
        """
        Generate a descriptive title for the meeting.

        Args:
            transcript: The full meeting transcript text.

        Returns:
            A short, descriptive meeting title.
        """
        transcript = " ".join(transcript.split()[:800])  # truncate for safety
        system_prompt = (
            "OUTPUT FORMAT: Plain text title only, max 10 words.\n"
            "RULES: No code. No explanation. No markdown. No quotes.\n"
            "TASK: Generate a short descriptive title for the meeting transcript below."
        )
        return self._chat(system_prompt, f"Meeting transcript:\n\n{transcript}")

    def answer_question(self, transcript: str, question: str) -> str:
        """
        Answer a question about the meeting based on the transcript.

        Args:
            transcript: The full meeting transcript text.
            question: The user's question about the meeting.

        Returns:
            An answer based on the transcript content.
        """
        transcript = " ".join(transcript.split()[:800])  # truncate for safety
        system_prompt = (
            "You are a professional meeting notes assistant. "
            "Answer the user's question based ONLY on the meeting transcript provided. "
            "If the answer is not in the transcript, say so clearly."
        )
        user_message = (
            f"Meeting transcript:\n\n{transcript}\n\n"
            f"Question: {question}"
        )
        return self._chat(system_prompt, user_message)

    def process_meeting(self, transcript: str) -> dict:
        """
        Run full processing pipeline on a transcript.

        Args:
            transcript: The full meeting transcript text.

        Returns:
            Dictionary with title, summary, key_points, and action_items.
        """
        return {
            "title": self.generate_title(transcript),
            "summary": self.generate_summary(transcript),
            "key_points": self.extract_key_points(transcript),
            "action_items": self.extract_action_items(transcript),
        }
