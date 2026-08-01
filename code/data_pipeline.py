"""
Data pipeline for the AI WhatsApp Message Router hackathon.

Loads CSV context files, pre-computes indexes for fast lookup, and builds
rich per-message context dictionaries for LLM routing decisions.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Default dataset location: sibling `dataset/` folder at repo root.
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Affinity score deltas keyed by reaction / interaction type.
EVENT_SCORES: dict[str, int] = {
    "replied": 2,
    "opened": 1,
    "dismissed": -1,
    "muted": -3,
    "reported": -5,
}


def _resolve_sender_id(row: pd.Series) -> Optional[str]:
    """
    Map a message row to its canonical sender identifier.

    Personal chats use sender_user_id; group chats use group_id;
    business chats use business_id.
    """
    conversation_type = row.get("conversation_type")
    if conversation_type == "personal":
        return row.get("sender_user_id") or None
    if conversation_type == "group":
        return row.get("group_id") or None
    if conversation_type == "business":
        return row.get("business_id") or None
    return None


def _extract_reactions(event_row: pd.Series) -> list[str]:
    """Convert binary event flags into a list of human-readable reaction labels."""
    reactions: list[str] = []
    if int(event_row.get("message_replied", 0) or 0) == 1:
        reactions.append("replied")
    if int(event_row.get("message_opened", 0) or 0) == 1:
        reactions.append("opened")
    if int(event_row.get("notification_dismissed", 0) or 0) == 1:
        reactions.append("dismissed")
    if int(event_row.get("muted_after_message", 0) or 0) == 1:
        reactions.append("muted")
    if int(event_row.get("message_reported", 0) or 0) == 1:
        reactions.append("reported")
    return reactions


def _score_reactions(reactions: list[str]) -> int:
    """Sum affinity deltas for a list of reaction labels."""
    return sum(EVENT_SCORES[reaction] for reaction in reactions)


def _clean_str(value: Any) -> str:
    """Normalize nullable string fields to plain strings."""
    if pd.isna(value) or value is None:
        return ""
    return str(value)


def _series_to_dict(row: pd.Series) -> dict[str, Any]:
    """Convert a DataFrame row to a JSON-friendly dictionary."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            try:
                result[key] = value.item()
            except (ValueError, AttributeError):
                result[key] = value
        else:
            result[key] = value
    return result


class ContextBuilder:
    """
    Loads WhatsApp routing context CSVs and exposes fast lookup helpers.

    All heavy joins and aggregations happen once at initialization (or on the
    first affinity/history build). Per-message context retrieval uses dict
    indexes only — no row-by-row DataFrame scans at request time.
    """

    def __init__(self, dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)

        # Raw tables loaded from CSV.
        self.messages: pd.DataFrame
        self.users: pd.DataFrame
        self.groups: pd.DataFrame
        self.group_members: pd.DataFrame
        self.business_accounts: pd.DataFrame
        self.user_business_history: pd.DataFrame
        self.message_history: pd.DataFrame
        self.message_events: pd.DataFrame

        # O(1) lookup indexes built after load.
        self._messages_by_id: dict[str, pd.Series] = {}
        self._users_by_id: dict[str, pd.Series] = {}
        self._groups_by_id: dict[str, pd.Series] = {}
        self._business_by_id: dict[str, pd.Series] = {}
        self._group_membership_by_key: dict[tuple[str, str], pd.Series] = {}
        self._business_history_by_key: dict[tuple[str, str], pd.Series] = {}
        self._events_by_user_message: dict[tuple[str, str], pd.Series] = {}

        # Precomputed affinity and historical evidence caches.
        self._affinity_scores: dict[tuple[str, str], int] = {}
        self._historical_evidence_index: dict[tuple[str, str], list[dict[str, Any]]] = {}

        self._load_dataframes()
        self._build_indexes()
        self.calculate_affinity_scores()
        self._build_historical_evidence_index()

    # ------------------------------------------------------------------
    # Loading and preprocessing
    # ------------------------------------------------------------------

    def _load_dataframes(self) -> None:
        """Load all required CSV files and fill missing values gracefully."""
        csv_names = [
            "messages",
            "users",
            "groups",
            "group_members",
            "business_accounts",
            "user_business_history",
            "message_history",
            "message_events",
        ]

        for name in csv_names:
            path = self.dataset_dir / f"{name}.csv"
            df = pd.read_csv(path, dtype=str, keep_default_na=True)

            # Preserve numeric columns where appropriate after string load.
            if name in {"messages", "message_history"}:
                df["forwarded_count"] = pd.to_numeric(
                    df["forwarded_count"], errors="coerce"
                ).fillna(0).astype(int)
                df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

            if name == "users":
                numeric_cols = [
                    "messages_opened_30d",
                    "messages_replied_30d",
                    "notifications_dismissed_30d",
                    "messages_reported_30d",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                df["do_not_disturb_window"] = df["do_not_disturb_window"].fillna("")

            if name == "groups":
                for col in ("member_count", "admin_count", "messages_30d"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                for col in ("group_name", "group_type"):
                    df[col] = df[col].fillna("")

            if name == "group_members":
                numeric_cols = [
                    "messages_sent_30d",
                    "messages_read_30d",
                    "replies_sent_30d",
                    "notifications_dismissed_30d",
                    "group_muted_by_user",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                df["role"] = df["role"].fillna("member")

            if name == "business_accounts":
                numeric_cols = [
                    "verified",
                    "account_age_days",
                    "messages_sent_30d",
                    "user_reports_30d",
                    "domain_used_by_sender_age_days",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                for col in (
                    "display_name",
                    "brand_name",
                    "category",
                    "official_domain",
                    "domain_used_by_sender",
                ):
                    df[col] = df[col].fillna("")

            if name == "user_business_history":
                numeric_cols = [
                    "allows_promotions",
                    "activity_count_180d",
                    "messages_opened_30d",
                    "messages_dismissed_30d",
                    "messages_replied_30d",
                ]
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                for col in ("why_user_knows_account", "last_activity_at", "last_reply_at"):
                    df[col] = df[col].fillna("")
                # promotions_opted_out_at stays nullable — empty means no opt-out.

            if name == "message_events":
                flag_cols = [
                    "message_opened",
                    "message_replied",
                    "notification_dismissed",
                    "muted_after_message",
                    "message_reported",
                ]
                for col in flag_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                df["reaction_time_minutes"] = pd.to_numeric(
                    df["reaction_time_minutes"], errors="coerce"
                )

            # Generic fill for remaining object columns.
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna("")

            setattr(self, name, df)

    def _build_indexes(self) -> None:
        """Materialize dict indexes for constant-time retrieval during routing."""
        self._messages_by_id = {
            row["message_id"]: row for _, row in self.messages.iterrows()
        }
        self._users_by_id = {
            row["user_id"]: row for _, row in self.users.iterrows()
        }
        self._groups_by_id = {
            row["group_id"]: row for _, row in self.groups.iterrows()
        }
        self._business_by_id = {
            row["business_id"]: row for _, row in self.business_accounts.iterrows()
        }

        self._group_membership_by_key = {
            (row["group_id"], row["user_id"]): row
            for _, row in self.group_members.iterrows()
        }
        self._business_history_by_key = {
            (row["user_id"], row["business_id"]): row
            for _, row in self.user_business_history.iterrows()
        }
        self._events_by_user_message = {
            (row["user_id"], row["message_id"]): row
            for _, row in self.message_events.iterrows()
        }

    # ------------------------------------------------------------------
    # Affinity scoring
    # ------------------------------------------------------------------

    def calculate_affinity_scores(self) -> dict[tuple[str, str], int]:
        """
        Compute Trust/Affinity scores for every (user_id, sender_id) pair.

        Sources:
          1. message_events.csv joined to message_history.csv (per-message reactions)
          2. user_business_history.csv (aggregated 30-day business engagement)

        Scoring rules:
          +2 replied, +1 opened, -1 dismissed, -3 muted, -5 reported

        Returns a dictionary keyed by (user_id, sender_id) for O(1) lookup.
        """
        scores: dict[tuple[str, str], int] = defaultdict(int)

        # --- Signal 1: historical message reactions -------------------
        # Vectorized sender resolution on message_history, then merge events.
        history = self.message_history.copy()
        history["sender_id"] = history.apply(_resolve_sender_id, axis=1)

        events_with_sender = self.message_events.merge(
            history[["message_id", "sender_id"]],
            on="message_id",
            how="inner",
        )

        for _, event_row in events_with_sender.iterrows():
            user_id = _clean_str(event_row["user_id"])
            sender_id = _clean_str(event_row["sender_id"])
            if not user_id or not sender_id:
                continue

            reactions = _extract_reactions(event_row)
            scores[(user_id, sender_id)] += _score_reactions(reactions)

        # --- Signal 2: user ↔ business relationship aggregates --------
        for _, rel_row in self.user_business_history.iterrows():
            user_id = _clean_str(rel_row["user_id"])
            business_id = _clean_str(rel_row["business_id"])
            if not user_id or not business_id:
                continue

            key = (user_id, business_id)
            scores[key] += int(rel_row["messages_replied_30d"]) * EVENT_SCORES["replied"]
            scores[key] += int(rel_row["messages_opened_30d"]) * EVENT_SCORES["opened"]
            scores[key] += int(rel_row["messages_dismissed_30d"]) * EVENT_SCORES["dismissed"]

            # Treat promotion opt-out as a strong negative trust signal.
            if _clean_str(rel_row.get("promotions_opted_out_at", "")):
                scores[key] += EVENT_SCORES["muted"]

        self._affinity_scores = dict(scores)
        return self._affinity_scores

    def get_affinity_score(self, user_id: str, sender_id: str) -> int:
        """Return the cached affinity score, defaulting to 0 when unknown."""
        return self._affinity_scores.get((user_id, sender_id), 0)

    # ------------------------------------------------------------------
    # Historical evidence
    # ------------------------------------------------------------------

    def _build_historical_evidence_index(self) -> None:
        """
        Pre-group message_history by (user_id, sender_id), newest first.

        Each bucket keeps up to 3 messages so get_historical_evidence()
        never scans the full history table at call time.
        """
        history = self.message_history.copy()
        history["sender_id"] = history.apply(_resolve_sender_id, axis=1)
        history = history.dropna(subset=["user_id", "sender_id"])
        history = history.sort_values("created_at", ascending=False)

        index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for _, row in history.iterrows():
            key = (_clean_str(row["user_id"]), _clean_str(row["sender_id"]))
            if len(index[key]) >= 3:
                continue

            message_id = _clean_str(row["message_id"])
            event_row = self._events_by_user_message.get((key[0], message_id))
            reactions = _extract_reactions(event_row) if event_row is not None else []

            index[key].append(
                {
                    "message_id": message_id,
                    "message_text": _clean_str(row.get("message_text", "")),
                    "created_at": row.get("created_at"),
                    "conversation_type": _clean_str(row.get("conversation_type", "")),
                    "media_type": _clean_str(row.get("media_type", "")),
                    "forwarded_count": int(row.get("forwarded_count", 0) or 0),
                    "reactions": reactions,
                    "affinity_delta": _score_reactions(reactions),
                }
            )

        self._historical_evidence_index = dict(index)

    def get_historical_evidence(self, user_id: str, sender_id: str) -> list[dict[str, Any]]:
        """
        Return the 1–3 most recent historical messages from sender_id to user_id.

        Each item includes message_id, message_text, and the user's reactions
        pulled from message_events.csv — critical for LLM evidence_message_ids.
        """
        evidence = self._historical_evidence_index.get((user_id, sender_id), [])
        # Return a shallow copy so callers cannot mutate the cache.
        return [dict(item) for item in evidence[:3]]

    # ------------------------------------------------------------------
    # Full message context assembly
    # ------------------------------------------------------------------

    def get_message_context(self, message_id: str) -> dict[str, Any]:
        """
        Build a deeply nested context dictionary for one incoming message.

        Combines raw message fields, user profile, group/business metadata,
        trust score, and recent interaction evidence — all via dict lookups.
        """
        message_row = self._messages_by_id.get(message_id)
        if message_row is None:
            raise KeyError(f"message_id '{message_id}' not found in messages.csv")

        user_id = _clean_str(message_row["user_id"])
        conversation_type = _clean_str(message_row["conversation_type"])
        sender_id = _resolve_sender_id(message_row)

        # --- Core message payload -------------------------------------
        message_context: dict[str, Any] = {
            "message_id": message_id,
            "user_id": user_id,
            "conversation_type": conversation_type,
            "created_at": message_row.get("created_at"),
            "message_text": _clean_str(message_row.get("message_text", "")),
            "media_type": _clean_str(message_row.get("media_type", "")),
            "media_id": _clean_str(message_row.get("media_id", "")),
            "forwarded_count": int(message_row.get("forwarded_count", 0) or 0),
            "sender_id": sender_id,
        }

        if conversation_type == "personal":
            message_context["sender_user_id"] = _clean_str(message_row.get("sender_user_id", ""))
        elif conversation_type == "group":
            message_context["group_id"] = _clean_str(message_row.get("group_id", ""))
            message_context["sender_user_id"] = _clean_str(message_row.get("sender_user_id", ""))
        elif conversation_type == "business":
            message_context["business_id"] = _clean_str(message_row.get("business_id", ""))

        # --- Receiving user profile -----------------------------------
        user_row = self._users_by_id.get(user_id)
        user_context: dict[str, Any] = {
            "user_id": user_id,
            "quiet_hours": user_row["do_not_disturb_window"] if user_row is not None else "",
            "profile": _series_to_dict(user_row) if user_row is not None else None,
        }

        # --- Group context (if applicable) ----------------------------
        group_context: Optional[dict[str, Any]] = None
        membership_context: Optional[dict[str, Any]] = None

        if conversation_type == "group":
            group_id = _clean_str(message_row.get("group_id", ""))
            group_row = self._groups_by_id.get(group_id)
            membership_row = self._group_membership_by_key.get((group_id, user_id))

            if group_row is not None:
                group_context = {
                    "group_id": group_id,
                    "info": _series_to_dict(group_row),
                }
            if membership_row is not None:
                membership_context = {
                    "group_id": group_id,
                    "user_id": user_id,
                    "role": _clean_str(membership_row.get("role", "member")),
                    "group_muted_by_user": bool(int(membership_row.get("group_muted_by_user", 0))),
                    "details": _series_to_dict(membership_row),
                }

        # --- Business context (if applicable) -------------------------
        business_context: Optional[dict[str, Any]] = None
        business_history_context: Optional[dict[str, Any]] = None

        if conversation_type == "business":
            business_id = _clean_str(message_row.get("business_id", ""))
            business_row = self._business_by_id.get(business_id)
            history_row = self._business_history_by_key.get((user_id, business_id))

            if business_row is not None:
                business_context = {
                    "business_id": business_id,
                    "verified": bool(int(business_row.get("verified", 0))),
                    "official_domain": _clean_str(business_row.get("official_domain", "")),
                    "domain_used_by_sender": _clean_str(business_row.get("domain_used_by_sender", "")),
                    "details": _series_to_dict(business_row),
                }
            if history_row is not None:
                business_history_context = {
                    "user_id": user_id,
                    "business_id": business_id,
                    "relationship": _series_to_dict(history_row),
                }

        # --- Trust / affinity and historical evidence -----------------
        affinity_score = self.get_affinity_score(user_id, sender_id or "")
        historical_evidence = (
            self.get_historical_evidence(user_id, sender_id)
            if sender_id
            else []
        )

        return {
            "message": message_context,
            "user": user_context,
            "group": group_context,
            "group_membership": membership_context,
            "business": business_context,
            "business_history": business_history_context,
            "trust": {
                "sender_id": sender_id,
                "affinity_score": affinity_score,
                "score_legend": EVENT_SCORES,
            },
            "historical_evidence": historical_evidence,
        }


if __name__ == "__main__":
    # Quick smoke test when run directly.
    builder = ContextBuilder()
    sample_id = builder.messages.iloc[0]["message_id"]
    context = builder.get_message_context(sample_id)
    print(f"Built context for {sample_id}")
    print(f"  conversation_type: {context['message']['conversation_type']}")
    print(f"  affinity_score:    {context['trust']['affinity_score']}")
    print(f"  evidence_count:    {len(context['historical_evidence'])}")
