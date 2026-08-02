"""
LLM-powered routing agent for the WhatsApp Message Notification Router.

Uses OpenAI chat completions (and Whisper for voice notes) to decide whether
each incoming message should notify, digest, or mute the user.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

import openai

# Media paths relative to the repo's dataset/ folder.
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
AUDIO_DIR = DEFAULT_DATASET_DIR / "media" / "audio"
IMAGE_DIR = DEFAULT_DATASET_DIR / "media" / "images"

# Safe fallback returned when routing fails (missing media, API error, bad JSON).
FALLBACK_ROUTE: dict[str, Any] = {
    "action": "digest",
    "message_type": "unknown",
    "reason": "Routing unavailable; defaulting to digest until the message can be reviewed.",
    "confidence": 0.0,
    "evidence_message_ids": "none",
}

SYSTEM_PROMPT = """You are an expert WhatsApp Message Notification Router.

## Goal
For each incoming WhatsApp message, decide how the receiving user should be notified:
- notify: interrupt the user now (urgent, time-sensitive, or personally important)
- digest: useful but low priority; safe to show later in a batch
- mute: suppress notification (repetitive, unwanted, low-value, spam, scam, or unsafe)

## Allowed message_type values
personal, urgent, event, payment, business_update, promotion, greeting, forward, spam, scam, unknown

## Routing rules
1. notify — Use when the message is urgent, time-bound, safety-related, a direct @mention with a deadline, a trusted admin/school/society update the user likely needs now, or a personal message requiring immediate response. Respect quiet hours: still notify for true emergencies, but lower confidence if outside active hours unless urgency is clear.

2. digest — Use for legitimate but non-urgent updates: routine business receipts, low-priority group chatter, informational content the user may read later, or messages with moderate trust but no immediate deadline.

3. mute — Use for repetitive promotions the user has dismissed before, chain forwards (especially high forwarded_count), scam/spam/phishing patterns, unverified business domains, messages similar to historically reported/dismissed/muted content, or content the user has opted out of. Risky scam/spam should be mute even if affinity is neutral.

## Personalization signals (use carefully)
- trust.affinity_score: higher = more trusted sender relationship; negative = prior dismissals, mutes, reports
- historical_evidence: past messages from the same sender and how the user reacted (replied, opened, dismissed, muted, reported). Cite relevant message_id values in evidence_message_ids when they support your decision.
- user.quiet_hours: do_not_disturb_window — factor into notify vs digest unless truly urgent
- group_membership.group_muted_by_user: user muted the group — prefer digest/mute unless urgent @mention
- business.verified and domain fields: mismatched or unverified domains increase scam risk → mute
- forwarded_count: high values (especially with promotional/scam patterns) → mute or digest

## Output format (CRITICAL)
Respond with ONLY a valid JSON object — no markdown, no prose outside JSON.
Required keys:
- action: string — one of notify, digest, mute
- message_type: string — one of the allowed message_type values
- reason: string — short human-readable explanation (1-2 sentences)
- confidence: float — between 0.0 and 1.0
- evidence_message_ids: string — semicolon-separated historical message IDs from historical_evidence that support your decision, or the literal string "none" if none apply

Example:
{"action":"mute","message_type":"scam","reason":"Unverified payment link matches prior reported phishing pattern.","confidence":0.91,"evidence_message_ids":"message_0028;message_0107"}
"""


class MessageRouter:
    """
    Routes WhatsApp messages using OpenAI models and structured JSON output.

    Expects OPENAI_API_KEY in the environment. Accepts nested context dicts
    produced by ContextBuilder.get_message_context().
    """

    def __init__(
        self,
        dataset_dir: Path | str = DEFAULT_DATASET_DIR,
        chat_model: str = "gpt-4o-mini",
        vision_model: str = "gpt-4o",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "media" / "audio"
        self.image_dir = self.dataset_dir / "media" / "images"
        self.chat_model = chat_model
        self.vision_model = vision_model
        self.client = openai.OpenAI()

    @staticmethod
    def encode_image(image_path: Path | str) -> str:
        """Read an image file from disk and return its base64-encoded string."""
        path = Path(image_path)
        with path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _resolve_image_path(self, media_id: str) -> Optional[Path]:
        """Locate an image file by media_id (.jpg first, then .png)."""
        if not media_id:
            return None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = self.image_dir / f"{media_id}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _resolve_audio_path(self, media_id: str) -> Optional[Path]:
        """Locate a voice note MP3 by media_id."""
        if not media_id:
            return None
        candidate = self.audio_dir / f"{media_id}.mp3"
        return candidate if candidate.is_file() else None

    def _image_mime_type(self, image_path: Path) -> str:
        """Return the data-URL MIME type for an image file extension."""
        ext = image_path.suffix.lower()
        if ext == ".png":
            return "image/png"
        return "image/jpeg"

    def _transcribe_voice(self, media_id: str) -> str:
        """
        Transcribe a voice note with Whisper.

        Returns transcript text, or an empty string if the file is missing
        or transcription fails.
        """
        audio_path = self._resolve_audio_path(media_id)
        if audio_path is None:
            return ""

        try:
            with audio_path.open("rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            return (transcription.text or "").strip()
        except (openai.OpenAIError, OSError, ValueError):
            return ""

    def _build_context_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare context for the prompt, optionally enriching voice messages
        with a Whisper transcript.
        """
        payload = json.loads(json.dumps(context, default=str))
        message = payload.get("message", {})
        media_type = message.get("media_type", "")

        if media_type == "voice":
            media_id = message.get("media_id", "")
            transcript = self._transcribe_voice(media_id)
            if transcript:
                existing_text = (message.get("message_text") or "").strip()
                if existing_text:
                    message["message_text"] = f"{existing_text}\n\n[Voice transcript]: {transcript}"
                else:
                    message["message_text"] = f"[Voice transcript]: {transcript}"
                message["voice_transcript"] = transcript
            else:
                message["voice_transcript"] = ""

        return payload

    def _parse_route_response(self, raw_content: str) -> dict[str, Any]:
        """Parse and normalize the model's JSON routing decision."""
        parsed = json.loads(raw_content)

        action = str(parsed.get("action", "digest")).lower()
        if action not in {"notify", "digest", "mute"}:
            action = "digest"

        message_type = str(parsed.get("message_type", "unknown")).lower()
        allowed_types = {
            "personal",
            "urgent",
            "event",
            "payment",
            "business_update",
            "promotion",
            "greeting",
            "forward",
            "spam",
            "scam",
            "unknown",
        }
        if message_type not in allowed_types:
            message_type = "unknown"

        reason = str(parsed.get("reason", FALLBACK_ROUTE["reason"]))
        confidence = float(parsed.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        evidence = parsed.get("evidence_message_ids", "none")
        evidence_str = "none" if evidence in (None, "", "none") else str(evidence)

        return {
            "action": action,
            "message_type": message_type,
            "reason": reason,
            "confidence": confidence,
            "evidence_message_ids": evidence_str,
        }

    def route_message(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Route a single message using full pipeline context.

        Handles voice transcription (Whisper), image vision input (base64),
        and returns a parsed routing JSON dict. On any failure, returns a
        safe digest/unknown fallback.
        """
        try:
            context_payload = self._build_context_payload(context)
            context_json = json.dumps(context_payload, indent=2, default=str)

            user_text = (
                "Analyze the incoming WhatsApp message using the context below "
                "and return your routing decision as JSON only.\n\n"
                f"Context:\n{context_json}"
            )

            user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]

            message = context.get("message", {})
            media_type = message.get("media_type", "")
            media_id = message.get("media_id", "")
            model = self.chat_model

            # Attach image for multimodal reasoning when present.
            if media_type == "image" and media_id:
                image_path = self._resolve_image_path(media_id)
                if image_path is not None:
                    try:
                        base64_image = self.encode_image(image_path)
                        mime_type = self._image_mime_type(image_path)
                        user_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            }
                        )
                        model = self.vision_model
                    except OSError:
                        # Missing/unreadable image — continue with text-only context.
                        pass

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content
            if not raw_content:
                return dict(FALLBACK_ROUTE)

            return self._parse_route_response(raw_content)

        except (openai.OpenAIError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return dict(FALLBACK_ROUTE)


if __name__ == "__main__":
    # Smoke test: requires OPENAI_API_KEY and an existing message context.
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run the router smoke test.")
    else:
        from data_pipeline import ContextBuilder

        builder = ContextBuilder()
        router = MessageRouter()
        sample_id = builder.messages.iloc[0]["message_id"]
        ctx = builder.get_message_context(sample_id)
        decision = router.route_message(ctx)
        print(json.dumps(decision, indent=2))
