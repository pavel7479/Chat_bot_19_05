from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from the_First_Agent.config.resource_paths import SEMANTIC_INTENTS_PATH


@dataclass(frozen=True, slots=True)
class PromptBlock:
    name: str
    required_intent_ids: frozenset[str]
    text: str
    section: str


class TopicPromptBlockCatalog:
    _BACKTICK_ID_RE = re.compile(r"`([a-z_]+)`")
    _JSON_INTENT_ID_RE = re.compile(r'"intent_id"\s*:\s*"([a-z_]+)"')

    def __init__(self) -> None:
        self._known_intent_ids = self._load_known_intent_ids()
        self._blocks = self._build_blocks()

    def blocks(self) -> tuple[PromptBlock, ...]:
        return self._blocks

    def known_intent_ids(self) -> frozenset[str]:
        return frozenset(self._known_intent_ids)

    def referenced_intent_ids(self, block: PromptBlock) -> frozenset[str]:
        referenced: set[str] = set()
        for pattern in (self._BACKTICK_ID_RE, self._JSON_INTENT_ID_RE):
            for match in pattern.findall(block.text):
                if match in self._known_intent_ids:
                    referenced.add(match)
        return frozenset(referenced)

    def effective_required_intent_ids(self, block: PromptBlock) -> frozenset[str]:
        return frozenset(set(block.required_intent_ids) | set(self.referenced_intent_ids(block)))

    def block_is_compatible(self, block: PromptBlock, selected_ids: set[str]) -> bool:
        return self.effective_required_intent_ids(block).issubset(selected_ids)

    def validate_block_definitions(self) -> list[str]:
        errors: list[str] = []
        for block in self._blocks:
            declared_unknown = sorted(set(block.required_intent_ids) - self._known_intent_ids)
            if declared_unknown:
                errors.append(f"{block.name}: unknown required_intent_ids {declared_unknown}")
            referenced = self.referenced_intent_ids(block)
            undeclared = sorted(set(referenced) - set(block.required_intent_ids))
            if undeclared:
                errors.append(
                    f"{block.name}: referenced intent_ids not declared in required_intent_ids {undeclared}"
                )
        return errors

    @staticmethod
    def _load_known_intent_ids() -> set[str]:
        raw = yaml.safe_load(SEMANTIC_INTENTS_PATH.read_text(encoding="utf-8")) or {}
        known: set[str] = set()
        for item in raw.get("intents", []):
            if not isinstance(item, dict):
                continue
            intent_id = str(item.get("intent", "")).strip()
            if intent_id:
                known.add(intent_id)
        return known

    @staticmethod
    def _build_blocks() -> tuple[PromptBlock, ...]:
        return (
            PromptBlock(
                name="tis_brand_names_for_calculation",
                required_intent_ids=frozenset({"tis_tariffs", "specific_brand_check", "brand_list_request"}),
                section="rules",
                text=(
                    "- Если последняя реплика перечисляет конкретные бренды в контексте TIS, "
                    "главная тема — `tis_tariffs`, а `specific_brand_check` может быть второй темой; "
                    "`brand_list_request` оставляй только для общего запроса списка брендов."
                ),
            ),
            PromptBlock(
                name="short_reply_context_resolution",
                required_intent_ids=frozenset({"legal_entity_purchase_flow", "physical_person_purchase", "demo_access", "company_services_info"}),
                section="rules",
                text="- Короткие ответы `да`, `нет`, `являюсь`, `yes`, `nope`, `пока изучаю` нужно связывать с последним вопросом бота и промежуточным пониманием, а не трактовать изолированно.",
            ),
            PromptBlock(
                name="general_pricing_dual",
                required_intent_ids=frozenset({"epc_tariffs", "tis_tariffs"}),
                section="rules",
                text=(
                    "- Общий pricing dual допустим только если клиент не указал `EPC`, не указал `TIS` "
                    "и в истории не обсуждается конкретный продукт."
                ),
            ),
            PromptBlock(
                name="purchase_and_pricing_dual_rule",
                required_intent_ids=frozenset({"purchase_ready", "epc_tariffs", "tis_tariffs"}),
                section="rules",
                text=(
                    "- Если клиент одновременно выражает готовность подключить доступ и спрашивает цену, "
                    "главной темой может остаться `purchase_ready`, а pricing intent допустим второй темой."
                ),
            ),
            PromptBlock(
                name="explicit_tis_priority",
                required_intent_ids=frozenset({"epc_tariffs", "tis_tariffs"}),
                section="rules",
                text="- Если последняя реплика явно содержит `TIS` или `тис`, выбирай `tis_tariffs`; не добавляй `epc_tariffs` только из-за общего pricing-контекста.",
            ),
            PromptBlock(
                name="explicit_epc_priority",
                required_intent_ids=frozenset({"epc_tariffs", "tis_tariffs"}),
                section="rules",
                text="- Если последняя реплика явно содержит `EPC` или `епс`, выбирай `epc_tariffs`; не добавляй `tis_tariffs` только из-за общего pricing-контекста.",
            ),
            PromptBlock(
                name="product_relation_or_difference_rule",
                required_intent_ids=frozenset({"product_relation_or_difference"}),
                section="rules",
                text=(
                    "- Вопросы о том, входит ли один продукт в другой, отдельно ли продукты подключаются, "
                    "чем они отличаются или что входит в состав продукта, относятся к `product_relation_or_difference`."
                ),
            ),
            PromptBlock(
                name="api_and_manager_dual_rule",
                required_intent_ids=frozenset({"api_integration", "human_operator_request"}),
                section="rules",
                text="- Если клиент одновременно просит API и помощь менеджера, допустимы `api_integration` + `human_operator_request`.",
            ),
            PromptBlock(
                name="catalog_list_vs_brand_list_rule",
                required_intent_ids=frozenset({"catalog_list_request", "brand_list_request", "specific_brand_check"}),
                section="rules",
                text=(
                    "- Вопросы о доступных каталогах, продуктах и типах каталогов относятся к `catalog_list_request`. "
                    "Вопросы о списке брендов относятся к `brand_list_request`. "
                    "Вопросы о наличии конкретной марки относятся к `specific_brand_check`."
                ),
            ),
            PromptBlock(
                name="payment_and_requisites_dual_rule",
                required_intent_ids=frozenset({"payment_without_details", "legal_entity_purchase_flow"}),
                section="rules",
                text="- Если клиент хочет оплатить, но не даёт ИНН или реквизиты, допустимы `payment_without_details` + `legal_entity_purchase_flow`.",
            ),
            PromptBlock(
                name="example_tis_only",
                required_intent_ids=frozenset({"tis_tariffs"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: Сколько стоит TIS?

Последняя реплика клиента:
Сколько стоит TIS?

Ответ:
{
  "intent_1": {
    "intent_id": "tis_tariffs",
    "score": 0.96,
    "reason": "Клиент явно спрашивает цену TIS."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_epc_only",
                required_intent_ids=frozenset({"epc_tariffs"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: Сколько стоит EPC?

Последняя реплика клиента:
Сколько стоит EPC?

Ответ:
{
  "intent_1": {
    "intent_id": "epc_tariffs",
    "score": 0.96,
    "reason": "Клиент явно спрашивает цену EPC."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_dual_pricing_general",
                required_intent_ids=frozenset({"epc_tariffs", "tis_tariffs"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: сколько стоит подписка

Последняя реплика клиента:
сколько стоит подписка

Ответ:
{
  "intent_1": {
    "intent_id": "epc_tariffs",
    "score": 0.95,
    "reason": "Это общий вопрос о цене без указания продукта."
  },
  "intent_2": {
    "intent_id": "tis_tariffs",
    "score": 0.72,
    "reason": "Общий вопрос о цене затрагивает и TIS."
  }
}""",
            ),
            PromptBlock(
                name="example_purchase_and_pricing",
                required_intent_ids=frozenset({"purchase_ready", "epc_tariffs", "tis_tariffs"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: хочу подключить доступ, что по стоимости

Последняя реплика клиента:
хочу подключить доступ, что по стоимости

Ответ:
{
  "intent_1": {
    "intent_id": "purchase_ready",
    "score": 0.94,
    "reason": "Клиент выражает готовность подключить доступ."
  },
  "intent_2": {
    "intent_id": "epc_tariffs",
    "score": 0.78,
    "reason": "Клиент одновременно спрашивает о цене без указания конкретного продукта."
  }
}""",
            ),
            PromptBlock(
                name="example_multi_device_access",
                required_intent_ids=frozenset({"multi_device_access"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: можно ли пользоваться нескольким людям

Последняя реплика клиента:
можно ли пользоваться нескольким людям

Ответ:
{
  "intent_1": {
    "intent_id": "multi_device_access",
    "score": 0.95,
    "reason": "Клиент спрашивает о совместном использовании доступа несколькими пользователями."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_catalog_list_request",
                required_intent_ids=frozenset({"catalog_list_request"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: какие каталоги у вас есть

Последняя реплика клиента:
какие каталоги у вас есть

Ответ:
{
  "intent_1": {
    "intent_id": "catalog_list_request",
    "score": 0.95,
    "reason": "Клиент просит перечислить доступные каталоги или продукты."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_specific_brand_check",
                required_intent_ids=frozenset({"specific_brand_check"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: есть ли каталог уаз

Последняя реплика клиента:
есть ли каталог уаз

Ответ:
{
  "intent_1": {
    "intent_id": "specific_brand_check",
    "score": 0.95,
    "reason": "Клиент спрашивает, поддерживается ли конкретная марка."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_product_relation_or_difference",
                required_intent_ids=frozenset({"product_relation_or_difference"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: Чем EPC отличается от TIS?

Последняя реплика клиента:
Чем EPC отличается от TIS?

Ответ:
{
  "intent_1": {
    "intent_id": "product_relation_or_difference",
    "score": 0.94,
    "reason": "Клиент спрашивает различие и связь продуктов EPC и TIS."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_abstract_catalog_difference",
                required_intent_ids=frozenset({"company_services_info", "product_relation_or_difference"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: чем отличается один каталог от другого

Последняя реплика клиента:
чем отличается один каталог от другого

Ответ:
{
  "intent_1": {
    "intent_id": "company_services_info",
    "score": 0.9,
    "reason": "Клиент абстрактно спрашивает о различиях между типами каталогов без явного упоминания EPC, TIS или EPC Full."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_tis_specific_brands_for_calculation",
                required_intent_ids=frozenset({"tis_tariffs", "specific_brand_check", "brand_list_request"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: Нужен TIS для Audi и BMW

Последняя реплика клиента:
Нужен TIS для Audi и BMW

Ответ:
{
  "intent_1": {
    "intent_id": "tis_tariffs",
    "score": 0.94,
    "reason": "Клиент просит расчёт или подбор TIS по конкретным брендам."
  },
  "intent_2": {
    "intent_id": "specific_brand_check",
    "score": 0.84,
    "reason": "В реплике указаны конкретные бренды."
  }
}""",
            ),
            PromptBlock(
                name="example_demo_legal_dual",
                required_intent_ids=frozenset({"demo_access", "legal_entity_purchase_flow"}),
                section="examples",
                text="""Пример:
Диалог:
бот: Мы можем предоставить демо только для автобизнеса. Вы представитель автобизнеса?
клиент: да, мы СТО

Последняя реплика клиента:
да, мы СТО

Ответ:
{
  "intent_1": {
    "intent_id": "demo_access",
    "score": 0.95,
    "reason": "Основная тема остаётся демо-доступом."
  },
  "intent_2": {
    "intent_id": "legal_entity_purchase_flow",
    "score": 0.89,
    "reason": "Клиент подтвердил важное условие для предоставления демо."
  }
}""",
            ),
            PromptBlock(
                name="example_short_reply_legal_status",
                required_intent_ids=frozenset({"legal_entity_purchase_flow"}),
                section="examples",
                text="""Пример:
Диалог:
бот: Вы юрлицо?
клиент: являюсь

Последняя реплика клиента:
являюсь

Ответ:
{
  "intent_1": {
    "intent_id": "legal_entity_purchase_flow",
    "score": 0.93,
    "reason": "Клиент коротко подтверждает юридический статус после вопроса бота."
  },
  "intent_2": null
}""",
            ),
            PromptBlock(
                name="example_demo_short_legal",
                required_intent_ids=frozenset({"demo_access", "legal_entity_purchase_flow"}),
                section="examples",
                text="""Пример:
Диалог:
бот: демо только для автобизнеса. Вы юрлицо?
клиент: yes, legal

Последняя реплика клиента:
yes, legal

Ответ:
{
  "intent_1": {
    "intent_id": "demo_access",
    "score": 0.94,
    "reason": "Активная тема диалога остаётся запросом на демо."
  },
  "intent_2": {
    "intent_id": "legal_entity_purchase_flow",
    "score": 0.88,
    "reason": "Клиент подтверждает юридический статус внутри активной темы демо."
  }
}""",
            ),
            PromptBlock(
                name="example_demo_short_physical",
                required_intent_ids=frozenset({"demo_access", "physical_person_purchase"}),
                section="examples",
                text="""Пример:
Диалог:
бот: демо только юрлицам. Вы ИП?
клиент: нет

Последняя реплика клиента:
нет

Ответ:
{
  "intent_1": {
    "intent_id": "demo_access",
    "score": 0.93,
    "reason": "Активная тема диалога остаётся запросом на демо."
  },
  "intent_2": {
    "intent_id": "physical_person_purchase",
    "score": 0.87,
    "reason": "Клиент отрицательно отвечает на вопрос о статусе и указывает на сценарий покупки не как юрлицо."
  }
}""",
            ),
            PromptBlock(
                name="example_api_manager_followup",
                required_intent_ids=frozenset({"api_integration", "human_operator_request"}),
                section="examples",
                text="""Пример:
Диалог:
клиент: нужен API
бот: можем обсудить интеграцию
клиент: соедините с менеджером

Последняя реплика клиента:
соедините с менеджером

Ответ:
{
  "intent_1": {
    "intent_id": "human_operator_request",
    "score": 0.92,
    "reason": "Клиент явно просит менеджера."
  },
  "intent_2": {
    "intent_id": "api_integration",
    "score": 0.84,
    "reason": "Просьба о менеджере относится к уже активной технической теме интеграции."
  }
}""",
            ),
            PromptBlock(
                name="example_discovery_not_ready",
                required_intent_ids=frozenset({"company_services_info", "purchase_ready"}),
                section="examples",
                text="""Пример:
Диалог:
бот: оформляем доступ?
клиент: нет, пока изучаю

Последняя реплика клиента:
нет, пока изучаю

Ответ:
{
  "intent_1": {
    "intent_id": "company_services_info",
    "score": 0.9,
    "reason": "Клиент не готов переходить к покупке и остаётся на стадии изучения сервиса."
  },
  "intent_2": null
}""",
            ),
        )
