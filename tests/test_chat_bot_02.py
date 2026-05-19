from __future__ import annotations

import json
import re
import sys
import unittest
from difflib import SequenceMatcher
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agents.policy.types import ResponseAction
from src.agents.response_services.pipeline_services import AnswerSynthesisService
from src.app.pricing_flow_state_service import PricingFlowStateService
from src.app.price_provider import PriceContext
from src.core.models import MandatoryMeaningBlock, PricingBrandItem, PricingBrandStatus, PricingFlowMode, PricingFlowState, SessionState
from src.domain.pricing import PriceCatalog
from src.main import build_app


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower().replace("ё", "е")).strip()


def _contains_any(text: str, needles: list[str]) -> bool:
    hay = _normalize(text)
    return any(_normalize(needle) in hay for needle in needles)


def _normalized_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=_normalize(left), b=_normalize(right)).ratio()


class _FakeLLMProvider:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def generate_json(self, prompt: str) -> str:
        return json.dumps(self._payload, ensure_ascii=False)


class ChatBotRegression02Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.app = build_app(cls.project_root)
        cls.snapshot_root = cls.project_root / "tests" / "_snapshots" / "test_chat_bot_02"
        cls.snapshot_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.app.clear_session(self._testMethodName)

    def _respond(self, message: str, *, turn_index: int | None = None):
        response = self.app.respond(session_id=self._testMethodName, user_query=message)
        debug = self.app.get_debug_trace(self._testMethodName)
        state = self.app._session.get_state(self._testMethodName)
        if turn_index is not None:
            self._snapshot_turn(turn_index, response, debug, state)
        return response, debug, state

    def _snapshot_turn(self, turn_index: int, response, debug: dict[str, object], state) -> None:
        target_dir = self.snapshot_root / self._testMethodName
        target_dir.mkdir(parents=True, exist_ok=True)
        state_path = target_dir / f"turn_{turn_index}_state.json"
        diagnostics_path = target_dir / f"turn_{turn_index}_diagnostics.json"
        state_payload = state.as_dict() if hasattr(state, "as_dict") else {}
        diagnostics_payload = {
            "response": {
                "topic_ids": list(getattr(response, "topic_ids", [])),
                "action_name": str(getattr(response, "action_name", "")),
                "answer_text": str(getattr(response, "answer_text", "")),
            },
            "debug": debug,
        }
        state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        diagnostics_path.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _topic_diagnostics(debug: dict[str, object]) -> dict[str, object]:
        value = debug.get("topic_result_diagnostics", {})
        return value if isinstance(value, dict) else {}

    def _agent_zero(self, debug: dict[str, object]) -> dict[str, object]:
        value = self._topic_diagnostics(debug).get("agent_zero_trace", {})
        return value if isinstance(value, dict) else {}

    def _shortlist(self, debug: dict[str, object]) -> dict[str, float]:
        shortlist = self._topic_diagnostics(debug).get("shortlist_trace", {})
        if not isinstance(shortlist, dict):
            return {}
        scores = shortlist.get("shortlist", shortlist)
        if not isinstance(scores, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in scores.items():
            try:
                result[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return result

    def _followup(self, debug: dict[str, object]) -> dict[str, object]:
        value = self._topic_diagnostics(debug).get("followup_trace", {})
        return value if isinstance(value, dict) else {}

    def _slot_trace(self, debug: dict[str, object]) -> dict[str, object]:
        value = self._topic_diagnostics(debug).get("slot_extraction_trace", {})
        return value if isinstance(value, dict) else {}

    def _answer_selection(self, debug: dict[str, object]) -> dict[str, object]:
        value = self._topic_diagnostics(debug).get("answer_selection_trace", {})
        return value if isinstance(value, dict) else {}

    def _assert_trace_presence(self, debug: dict[str, object]) -> None:
        diagnostics = self._topic_diagnostics(debug)
        for key in ("agent_zero_trace", "shortlist_trace", "slot_extraction_trace", "followup_trace"):
            self.assertIn(key, diagnostics, f"Missing diagnostics[{key}] in debug trace")

    def _assert_answer_selection_trace_contract(self, debug: dict[str, object]) -> None:
        diagnostics = self._topic_diagnostics(debug)
        self.assertIn("answer_selection_trace", diagnostics, "Missing diagnostics[answer_selection_trace] in debug trace")
        trace = diagnostics.get("answer_selection_trace", {})
        self.assertIsInstance(trace, dict, f"Expected answer_selection_trace dict, got {type(trace)!r}")

    def _assert_action(self, response, expected_action: str) -> None:
        self.assertEqual(
            response.action_name,
            expected_action,
            f"Expected action={expected_action}, got action={response.action_name}, answer={response.answer_text}",
        )

    def _assert_topic(self, response, expected_topic: str) -> None:
        self.assertIn(
            expected_topic,
            response.topic_ids,
            f"Expected topic `{expected_topic}`, got topics={response.topic_ids}, answer={response.answer_text}",
        )

    def test_01_greeting_via_agent_zero(self) -> None:
        response, debug, state = self._respond("здорова", turn_index=1)
        agent_zero = self._agent_zero(debug)

        self.assertEqual(agent_zero.get("turn_type"), "greeting")
        self._assert_action(response, "greeting_once")
        self.assertTrue(_contains_any(response.answer_text, ["здравствуйте"]))
        self.assertFalse(_contains_any(response.answer_text, ["актуальные данные", "стабильная работа"]))
        self.assertTrue(state.greeted)
        self.assertEqual(self._topic_diagnostics(debug).get("classifier_source"), "dialog_act_router")
        self.assertTrue(self._topic_diagnostics(debug).get("shortlist_trace", {}).get("skipped"))
        self._assert_trace_presence(debug)

    def test_02_greeting_not_repeated_second_time(self) -> None:
        self._respond("здорова", turn_index=1)
        response, debug, state = self._respond("привет", turn_index=2)

        self.assertTrue(state.greeted)
        self.assertNotEqual(
            response.action_name,
            "greeting_once",
            f"Second greeting must not trigger greeting_once again, answer={response.answer_text}",
        )
        self.assertLessEqual(response.answer_text.count("Здравствуйте"), 1)
        self._assert_trace_presence(debug)

    def test_03_service_continuation_anti_repeat(self) -> None:
        self._respond("здорова", turn_index=1)
        second, second_debug, _ = self._respond("ну попробуй", turn_index=2)
        third, third_debug, _ = self._respond("рассказывай", turn_index=3)

        self._assert_action(second, "company_services")
        self._assert_action(third, "company_services")
        second_fact_ids = list(second_debug.get("answer_block", {}).get("used_fact_ids", []))
        third_fact_ids = list(third_debug.get("answer_block", {}).get("used_fact_ids", []))
        self.assertNotEqual(
            second_fact_ids,
            third_fact_ids,
            f"Service progression must move to new facts, got turn2={second_fact_ids}, turn3={third_fact_ids}",
        )
        self.assertLess(
            _normalized_similarity(second.answer_text, third.answer_text),
            0.7,
            f"Service answers repeat too closely: turn2={second.answer_text} | turn3={third.answer_text}",
        )
        self._assert_trace_presence(second_debug)
        self._assert_trace_presence(third_debug)

    def test_04_yes_continues_service(self) -> None:
        self._respond("здорова", turn_index=1)
        second, second_debug, _ = self._respond("ну попробуй", turn_index=2)
        third, third_debug, _ = self._respond("рассказывай", turn_index=3)
        fourth, fourth_debug, _ = self._respond("да", turn_index=4)

        agent_zero = self._agent_zero(fourth_debug)
        used_2 = list(second_debug.get("answer_block", {}).get("used_fact_ids", []))
        used_3 = list(third_debug.get("answer_block", {}).get("used_fact_ids", []))
        used_4 = list(fourth_debug.get("answer_block", {}).get("used_fact_ids", []))
        diagnostics = self._topic_diagnostics(fourth_debug)

        self.assertEqual(agent_zero.get("turn_type"), "service_discovery_continue")
        self.assertEqual(diagnostics.get("classifier_source"), "dialog_act_router")
        self.assertTrue(diagnostics.get("shortlist_trace", {}).get("skipped"))
        self.assertNotEqual(fourth.answer_text, third.answer_text)
        self.assertNotIn(
            used_4,
            [used_2, used_3],
            f"Turn 4 reused prior service fact ids: turn2={used_2}, turn3={used_3}, turn4={used_4}",
        )
        self._assert_trace_presence(fourth_debug)

    def test_05_pricing_flow_keeps_tis_context(self) -> None:
        first, first_debug, first_state = self._respond("сколько стоит подписка", turn_index=1)
        second, second_debug, _ = self._respond("mers, ford, автоваз, Captiva и РСД-10", turn_index=2)

        self._assert_action(first, "pricing_summary")
        self.assertTrue(_contains_any(first.answer_text, ["epc full"]))
        self.assertEqual(first_state.active_pricing_flow, "tis")
        self.assertEqual(self._agent_zero(second_debug).get("turn_type"), "brand_list_for_tis")
        self._assert_action(second, "tis_tariffs")
        self.assertNotEqual(
            second.action_name,
            "brand_availability",
            f"Brand list for TIS must stay in pricing flow, answer={second.answer_text}",
        )
        self._assert_trace_presence(first_debug)
        self._assert_trace_presence(second_debug)

    def test_06_state_keeps_full_brand_list(self) -> None:
        self._respond("сколько стоит подписка", turn_index=1)
        _, debug, state = self._respond("mers, ford, автоваз, Captiva и РСД-10", turn_index=2)
        slot_trace = self._slot_trace(debug)
        slots = slot_trace.get("slots", {}) if isinstance(slot_trace.get("slots", {}), dict) else {}

        self.assertEqual(
            slots.get("raw_brand_mentions"),
            ["mers", "ford", "автоваз", "Captiva", "РСД-10"],
            f"Unexpected raw brand mentions: {slots.get('raw_brand_mentions')}",
        )
        self.assertEqual(state.recognized_brands, ["ford", "lada"])
        self.assertEqual(state.unknown_brand_mentions, ["Mers", "Captiva", "РСД-10"])
        self.assertNotEqual(
            state.last_mentioned_brand,
            "lada",
            "State must not collapse multi-brand pricing flow to one last_mentioned_brand only.",
        )
        self._assert_trace_presence(debug)

    def test_07_remaining_brands_uses_state(self) -> None:
        self._respond("сколько стоит подписка", turn_index=1)
        self._respond("mers, ford, автоваз, Captiva и РСД-10", turn_index=2)
        self._respond("отдельно", turn_index=3)
        response, debug, _ = self._respond("а остальные", turn_index=4)

        self.assertEqual(self._agent_zero(debug).get("turn_type"), "pricing_followup")
        self.assertEqual(self._followup(debug).get("followup_type"), "remaining_brands_followup")
        self._assert_action(response, "tis_tariffs")
        self.assertFalse(_contains_any(response.answer_text, ["уточните бренды", "уточните стоимость"]))
        self.assertTrue(
            _contains_any(
                response.answer_text,
                ["не распознал", "цена tis в текущем прайсе не указана", "отдельно проверю"],
            ),
            f"Remaining-brands answer did not use stateful pricing info: {response.answer_text}",
        )
        self._assert_trace_presence(debug)

    def test_08_clarification_pushback_no_cost_loop(self) -> None:
        self._respond("сколько стоит подписка", turn_index=1)
        self._respond("mers, ford, автоваз, Captiva и РСД-10", turn_index=2)
        self._respond("отдельно", turn_index=3)
        self._respond("а остальные", turn_index=4)
        response, debug, _ = self._respond("что уточнить", turn_index=5)

        self.assertEqual(self._agent_zero(debug).get("turn_type"), "clarification_pushback")
        self.assertFalse(_contains_any(response.answer_text, ["уточните стоимость"]))
        self.assertTrue(
            _contains_any(response.answer_text, ["не распознал", "не указана", "менеджер", "специалист"]),
            f"Pushback answer must explain missing info or offer manager, got: {response.answer_text}",
        )
        self._assert_trace_presence(debug)

    def test_09_human_operator_action_cannot_return_pricing_clarification(self) -> None:
        self._respond("сколько стоит подписка", turn_index=1)
        self._respond("mers, ford, автоваз, Captiva и РСД-10", turn_index=2)
        self._respond("отдельно", turn_index=3)
        self._respond("а остальные", turn_index=4)
        self._respond("что уточнить", turn_index=5)
        response, debug, _ = self._respond("это ты мне уточни", turn_index=6)

        answer_selection = self._answer_selection(debug)
        self._assert_action(response, "human_operator")
        self.assertFalse(_contains_any(response.answer_text, ["уточните стоимость", "уточните тариф", "напишите бренды"]))
        self.assertNotEqual(answer_selection.get("selected_source"), "llm")
        self.assertIn(
            answer_selection.get("final_answer_source"),
            {"deterministic", "safe_default"},
            f"Unexpected final answer source for human_operator: {answer_selection}",
        )
        self._assert_trace_presence(debug)
        self._assert_answer_selection_trace_contract(debug)

    def test_10_validator_rejects_action_mismatch(self) -> None:
        service = AnswerSynthesisService()
        llm_provider = _FakeLLMProvider({"answer_text": "Пожалуйста, уточните стоимость TIS для бренда."})
        selected_action = ResponseAction(name="human_operator", primary_topic="human_operator_request")
        deterministic_answer = "Подключу менеджера: он поможет разобрать вопрос и свяжется с вами по дальнейшим шагам."
        answer_text, _, trace = service._generate_answer_text(
            llm_provider=llm_provider,
            answer_prompt="test",
            selected_action=selected_action,
            price_context=PriceContext(),
            deterministic_answer=deterministic_answer,
            fallback_answer="",
            mandatory_blocks=[
                MandatoryMeaningBlock(
                    key="human_operator_handoff",
                    required_phrases=["менеджер"],
                    semantic_tags=["human_operator_handoff"],
                )
            ],
            answerability_status="ok",
        )

        self.assertFalse(trace.get("action_compatible"))
        self.assertNotEqual(trace.get("selected_source"), "llm")
        self.assertIn(trace.get("final_answer_source"), {"deterministic", "safe_default"})
        self.assertTrue(_contains_any(answer_text, ["менеджер", "специалист", "свяжется"]))

    def test_11_followup_resolver_runs_for_dialog_act_pipeline(self) -> None:
        self._respond("Есть ли у вас каталог на Haval Dargo", turn_index=1)
        response, debug, _ = self._respond("подскажи", turn_index=2)

        followup = self._followup(debug)
        self._assert_action(response, "tis_tariffs")
        self.assertTrue(followup.get("is_followup"))
        self.assertEqual(followup.get("followup_type"), "brand_price_followup")
        self.assertEqual(followup.get("inherited_brand"), "haval")
        self.assertNotEqual(followup.get("skipped"), True)

    def test_12_dialog_act_pipeline_preserves_brand_price_followup(self) -> None:
        self._respond("есть ли каталог уаз", turn_index=1)
        response, debug, _ = self._respond("только уаз", turn_index=2)

        followup = self._followup(debug)
        self.assertTrue(any(topic in response.topic_ids for topic in ["partial_catalog_request", "tis_tariffs", "specific_brand_check"]))
        self.assertTrue(followup.get("is_followup"))
        self.assertEqual(followup.get("followup_type"), "brand_price_followup")
        self.assertEqual(followup.get("inherited_brand"), "uaz")
        self.assertNotEqual(followup.get("skipped"), True)

    def test_13_followup_trace_not_fake_for_dialog_act(self) -> None:
        self._respond("Есть ли у вас каталог на Haval Dargo", turn_index=1)
        _, debug, _ = self._respond("подскажи", turn_index=2)

        followup = self._followup(debug)
        self.assertNotEqual(followup.get("skipped"), True)
        self.assertIn("reason", followup)
        self.assertIn("state_before", followup)


class PricingFlowDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.price_catalog = PriceCatalog(cls.project_root / "src" / "config" / "prices.yaml")
        cls.price_provider = PriceProvider(cls.price_catalog)
        cls.pricing_flow_service = PricingFlowStateService(cls.project_root / "src" / "config" / "brands.yaml")

    def test_missing_price_brand_never_becomes_unresolved(self) -> None:
        flow = PricingFlowState(
            active=True,
            product="tis",
            mode=PricingFlowMode.ALL,
            brand_items=[
                PricingBrandItem(
                    canonical_brand="haval",
                    display_name="Haval",
                    raw_surface="haval",
                    recognized=True,
                    status=PricingBrandStatus.MISSING_PRICE,
                )
            ],
        )

        price_context = self.price_provider.build_for_pricing_flow(
            required_price_blocks=["tis"],
            pricing_flow=flow,
            followup_trace={},
        )

        self.assertEqual(price_context.missing_tis_price_brands, ["haval"])
        self.assertEqual(price_context.rendered_unresolved_surfaces, [])
        self.assertNotIn("Haval", price_context.unknown_brand_mentions)

    def test_pending_queue_contains_entities_not_surfaces(self) -> None:
        flow = PricingFlowState(
            active=True,
            product="tis",
            mode=PricingFlowMode.ALL,
            brand_items=[
                PricingBrandItem(
                    canonical_brand="ford",
                    display_name="Ford",
                    raw_surface="ford",
                    recognized=True,
                    status=PricingBrandStatus.MISSING_PRICE,
                )
            ],
        )

        derived = self.pricing_flow_service._sync_derived_fields(flow)

        self.assertEqual(len(derived.brand_items), 1)
        self.assertEqual(derived.brand_items[0].canonical_brand, "ford")
        self.assertTrue(derived.brand_items[0].recognized)
        self.assertEqual(derived.unknown_brand_mentions, [])
        self.assertEqual(derived.missing_price_brands, ["ford"])

    def test_explain_unresolved_does_not_include_recognized_brand(self) -> None:
        flow = PricingFlowState(
            active=True,
            product="tis",
            mode=PricingFlowMode.EXPLAIN_UNRESOLVED,
            brand_items=[
                PricingBrandItem(
                    canonical_brand="ford",
                    display_name="Ford",
                    raw_surface="ford",
                    recognized=True,
                    status=PricingBrandStatus.MISSING_PRICE,
                ),
                PricingBrandItem(
                    canonical_brand="",
                    display_name="Mers",
                    raw_surface="mers",
                    recognized=False,
                    status=PricingBrandStatus.UNRESOLVED,
                ),
            ],
        )

        price_context = self.price_provider.build_for_pricing_flow(
            required_price_blocks=["tis"],
            pricing_flow=flow,
            followup_trace={},
        )
        joined = "\n".join(price_context.price_lines).lower()

        self.assertIn("ford", joined)
        self.assertIn("не указана", joined)
        self.assertIn("mers не распознан", joined)
        self.assertNotIn("ford не распознан", joined)

    def test_remaining_only_uses_entities(self) -> None:
        flow = PricingFlowState(
            active=True,
            product="tis",
            mode=PricingFlowMode.REMAINING_ONLY,
            brand_items=[
                PricingBrandItem(
                    canonical_brand="lada",
                    display_name="Lada",
                    raw_surface="автоваз",
                    recognized=True,
                    has_price=True,
                    processed=True,
                    status=PricingBrandStatus.PRICED,
                ),
                PricingBrandItem(
                    canonical_brand="ford",
                    display_name="Ford",
                    raw_surface="ford",
                    recognized=True,
                    has_price=False,
                    processed=False,
                    status=PricingBrandStatus.MISSING_PRICE,
                ),
                PricingBrandItem(
                    canonical_brand="",
                    display_name="Captiva",
                    raw_surface="Captiva",
                    recognized=False,
                    processed=False,
                    status=PricingBrandStatus.UNRESOLVED,
                ),
            ],
        )

        price_context = self.price_provider.build_for_pricing_flow(
            required_price_blocks=["tis"],
            pricing_flow=flow,
            followup_trace={},
        )
        joined = "\n".join(price_context.price_lines).lower()

        self.assertIn("ford", joined)
        self.assertIn("captiva", joined)
        self.assertNotIn("lada", joined)

    def test_haval_missing_price_flow(self) -> None:
        session = PricingFlowState(
            active=True,
            product="tis",
            mode=PricingFlowMode.ALL,
            brand_items=[
                PricingBrandItem(
                    canonical_brand="haval",
                    display_name="Haval",
                    raw_surface="haval",
                    recognized=True,
                    status=PricingBrandStatus.MISSING_PRICE,
                )
            ],
        )

        price_context = self.price_provider.build_for_pricing_flow(
            required_price_blocks=["tis"],
            pricing_flow=session,
            followup_trace={"is_followup": True, "followup_type": "brand_price_followup"},
        )
        state = self.pricing_flow_service._sync_derived_fields(session)
        session_state = SessionState(
            active_pricing_flow="tis",
            pricing_requested_product="tis",
            pricing_mode="all",
            pricing_flow=state.as_dict(),
            recognized_brands=["haval"],
            missing_price_brands=["haval"],
            requested_brands=["haval"],
        )
        patch = self.pricing_flow_service.apply_response_event(
            state_before_response=session_state,
            response_action="tis_tariffs",
            answer_block={"price_context": price_context.as_dict()},
        ).to_session_patch()

        self.assertEqual(price_context.missing_tis_price_brands, ["haval"])
        self.assertIn("epc", price_context.fallback_price_blocks)
        self.assertEqual(price_context.rendered_unresolved_surfaces, [])
        self.assertEqual(patch["missing_price_brands"], ["haval"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# cd /root/project/Chat_bot && /root/project/.venv/bin/python -m unittest -v tests.test_chat_bot_02
