from __future__ import annotations

from pathlib import Path


FIRST_AGENT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_INTENTS_PATH = FIRST_AGENT_ROOT / "config" / "semantic_intents.yaml"
CONTEXT_SIGNAL_RULES_PATH = FIRST_AGENT_ROOT / "config" / "context_signal_rules.yaml"
TOPIC_CLASSIFIER_PROMPT_PATH = FIRST_AGENT_ROOT / "prompts" / "topic_classifier_prompt.txt"
