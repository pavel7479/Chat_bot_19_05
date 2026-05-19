from __future__ import annotations

from the_First_Agent.prompting.topic_prompt_block_catalog import TopicPromptBlockCatalog


class TopicPromptSectionsBuilder:
    def __init__(self, block_catalog: TopicPromptBlockCatalog | None = None) -> None:
        self._block_catalog = block_catalog or TopicPromptBlockCatalog()

    def build_rules_text(self, allowed_intent_ids: list[str] | set[str]) -> str:
        selected_ids = set(allowed_intent_ids)
        rules: list[str] = []
        for block in self._block_catalog.blocks():
            if block.section != "rules":
                continue
            if not self._block_catalog.block_is_compatible(block, selected_ids):
                continue
            rules.append(block.text)
        return "\n".join(rules)

    def build_examples_text(self, allowed_intent_ids: list[str] | set[str]) -> str:
        selected_ids = set(allowed_intent_ids)
        rendered: list[str] = []
        for block in self._block_catalog.blocks():
            if block.section != "examples":
                continue
            if not self._block_catalog.block_is_compatible(block, selected_ids):
                continue
            rendered.append(block.text)
        return "\n\n".join(rendered)
