"""Shared lifecycle-event helpers for the pure LangChain-Core callback adapter.

Single home for: envelope builders (wrapping the base SDK's ``activity_started``/
``activity_completed`` factories with ``extra`` fields), human-turn prompt
extraction, in-place redaction application, and ``__openbox`` activity-input
enrichment. Phase 3 (middleware rebuild) and Phase 5 (LangGraph LLM ownership,
another repo) both consume this module rather than re-extracting the logic.

Pure helpers only — no gate calls, no bridge state, no callback wiring.

Split into ``lifecycle_events_envelopes.py`` (envelope builders) and
``lifecycle_events_redaction.py`` (prompt extraction, redaction, enrichment)
to stay under 200 lines per file; this module is the single import surface.
"""

from __future__ import annotations

from openbox_langchain.lifecycle_events_envelopes import (
    build_activity_completed,
    build_activity_started,
)
from openbox_langchain.lifecycle_events_redaction import (
    apply_redaction_to_messages,
    enrich_activity_input,
    extract_human_turn_prompt,
)

__all__ = [
    "apply_redaction_to_messages",
    "build_activity_completed",
    "build_activity_started",
    "enrich_activity_input",
    "extract_human_turn_prompt",
]
