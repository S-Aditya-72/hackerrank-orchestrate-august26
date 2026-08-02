"""
Main orchestration script for the WhatsApp Message Notification Router.

Loads context for every incoming message, routes each one through the LLM
agent, and writes predictions to dataset/output.csv for hackathon submission.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_pipeline import ContextBuilder
from router_agent import MessageRouter

# Set to an integer (e.g. 10) to process only the first N messages during testing.
# Set to None to run the full dataset.
LIMIT: int | None = None

# Paths relative to this file (code/) and the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
MESSAGES_CSV = DATASET_DIR / "messages.csv"
OUTPUT_CSV = DATASET_DIR / "output.csv"

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

# Per-row fallback when context building or routing raises an exception.
PIPELINE_FALLBACK = {
    "action": "digest",
    "message_type": "unknown",
    "reason": "Pipeline error",
    "confidence": 0.0,
    "evidence_message_ids": "none",
}


def main() -> None:
    start_time = time.time()

    print("Initializing context builder...")
    builder = ContextBuilder(dataset_dir=DATASET_DIR)
    builder.calculate_affinity_scores()

    print("Initializing message router...")
    router = MessageRouter(dataset_dir=DATASET_DIR)

    print(f"Loading messages from {MESSAGES_CSV}...")
    messages_df = pd.read_csv(MESSAGES_CSV)
    message_ids = messages_df["message_id"].tolist()

    if LIMIT is not None:
        message_ids = message_ids[:LIMIT]
        print(f"LIMIT={LIMIT} — processing first {len(message_ids)} message(s) only.")

    results: list[dict[str, object]] = []

    for message_id in tqdm(message_ids, desc="Routing messages", unit="msg"):
        try:
            context = builder.get_message_context(message_id)
            decision = router.route_message(context)

            row = {
                "message_id": message_id,
                "action": decision.get("action", PIPELINE_FALLBACK["action"]),
                "message_type": decision.get("message_type", PIPELINE_FALLBACK["message_type"]),
                "reason": decision.get("reason", PIPELINE_FALLBACK["reason"]),
                "confidence": decision.get("confidence", PIPELINE_FALLBACK["confidence"]),
                "evidence_message_ids": decision.get(
                    "evidence_message_ids", PIPELINE_FALLBACK["evidence_message_ids"]
                ),
            }
        except Exception as exc:
            print(f"\nWarning: failed to route {message_id}: {exc}")
            row = {"message_id": message_id, **PIPELINE_FALLBACK}

        results.append(row)

    print(f"Writing {len(results)} prediction(s) to {OUTPUT_CSV}...")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s. Output saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
