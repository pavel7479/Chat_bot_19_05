from __future__ import annotations

import sys
from pathlib import Path

import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.agents.intent_agent import IntentAgent
from src.agents.response_agent import ResponseAgent
from src.app.chatbot import ChatBotOrchestrator
from src.app.answer_composer import AnswerComposer
from src.app.brand_alias_resolver import TurnBrandAliasResolver
from src.app.dialog_act_router import DialogActRouter
from src.app.knowledge_retriever import KnowledgeRetriever
from src.app.price_provider import PriceProvider
from src.app.pricing_flow_state_service import PricingFlowStateService
from src.app.product_resolver import ProductResolver
from src.app.slot_extraction_service import SlotExtractionService
from src.app.state_update_service import StateUpdateService
from src.app.telemetry_service import TelemetryService
from src.config.loader import ConfigLoader
from src.llm.provider_factory import LLMProviderFactory
from src.logging_system.logger import StructuredLoggerFactory
from src.prompting.prompt_manager import PromptManager
from src.retrieval.fact_repository import FactRepository
from src.domain.brands import BrandAliasResolver as DomainBrandAliasResolver
from src.domain.pricing import PriceCatalog
from src.session.session_manager import SessionManager
from the_First_Agent.Agent_Zero.context_understanding_agent import ContextUnderstandingAgent
from the_First_Agent.catalog.topic_catalog import TopicCatalog
from the_First_Agent.catalog.topic_shortlist_builder import TopicShortlistBuilder
from the_First_Agent.config.resource_paths import CONTEXT_SIGNAL_RULES_PATH, SEMANTIC_INTENTS_PATH, TOPIC_CLASSIFIER_PROMPT_PATH
from the_First_Agent.context.context_signal_extractor import ContextSignalExtractor
from the_First_Agent.followup.followup_resolver import FollowupResolver
from the_First_Agent.orchestrator.topic_classifier import TopicClassifier
from the_First_Agent.prompting.topic_prompt_sections_builder import TopicPromptSectionsBuilder

DEFAULT_CONFIG_PATH = "config.yaml"


def build_app(project_root: Path, config_path: str = DEFAULT_CONFIG_PATH) -> ChatBotOrchestrator:
    config = ConfigLoader(project_root / config_path).load()
    logger = StructuredLoggerFactory.create("chat_bot", config.logging, project_root)

    llm_provider = LLMProviderFactory.create(config.llm)
    topic_catalog = TopicCatalog(SEMANTIC_INTENTS_PATH)
    topic_shortlist_builder = TopicShortlistBuilder(topic_catalog.topics, top_k=8)
    topic_prompt_sections_builder = TopicPromptSectionsBuilder()
    session_manager = SessionManager(config.session.max_history_messages)
    prompt_manager = PromptManager(
        project_root,
        str(TOPIC_CLASSIFIER_PROMPT_PATH.relative_to(project_root)),
        config.paths.answer_generator_prompt,
    )
    context_understanding_agent = ContextUnderstandingAgent(
        llm=llm_provider,
        prompt_path=project_root / config.paths.context_understanding_prompt,
        logger=logger,
    )
    context_signal_extractor = ContextSignalExtractor(CONTEXT_SIGNAL_RULES_PATH)
    topic_classifier = TopicClassifier(
        llm_provider=llm_provider,
        topic_ids=set(topic_catalog.topics.keys()),
        topic_titles_by_id=topic_catalog.title_map(),
        intents_config_path=project_root / config.paths.intents_config_file,
        brands_file_path=project_root / config.paths.brands_file,
    )
    intent_agent = IntentAgent(
        topic_catalog=topic_catalog,
        topic_shortlist_builder=topic_shortlist_builder,
        topic_classifier=topic_classifier,
        prompt_manager=prompt_manager,
        topic_prompt_sections_builder=topic_prompt_sections_builder,
        context_understanding_agent=context_understanding_agent,
        context_signal_extractor=context_signal_extractor,
        followup_resolver=FollowupResolver(),
    )
    price_catalog = PriceCatalog(project_root / config.paths.prices_file)
    response_agent = ResponseAgent(
        llm_provider=llm_provider,
        prompt_manager=prompt_manager,
        brands_file_path=project_root / config.paths.brands_file,
        facts_file_path=project_root / "src/config/facts.yaml",
        response_policy_file_path=project_root / config.paths.response_policy_file,
        prices_file_path=project_root / config.paths.prices_file,
        response_fact_map_file_path=project_root / "src/config/response_fact_map.yaml",
        min_evidence_score=config.retrieval.min_evidence_score,
        min_evidence_hits=config.retrieval.min_evidence_hits,
    )
    knowledge_retriever = KnowledgeRetriever(
        fact_repository=FactRepository(project_root / "src/config/facts.yaml"),
        max_facts_per_turn=8,
    )
    brand_alias_resolver = TurnBrandAliasResolver(
        brand_resolver=DomainBrandAliasResolver(project_root / config.paths.brands_file),
    )
    pricing_flow_state_service = PricingFlowStateService(project_root / config.paths.brands_file)
    answer_composer = AnswerComposer(
        price_provider=PriceProvider(price_catalog=price_catalog),
        pricing_flow_state_service=pricing_flow_state_service,
        max_facts_in_context=5,
    )
    slot_extraction_service = SlotExtractionService(project_root / config.paths.brands_file)
    product_resolver = ProductResolver(project_root / config.paths.brands_file)

    return ChatBotOrchestrator(
        config=config,
        project_root=project_root,
        intent_agent=intent_agent,
        response_agent=response_agent,
        knowledge_retriever=knowledge_retriever,
        brand_alias_resolver=brand_alias_resolver,
        answer_composer=answer_composer,
        session_manager=session_manager,
        logger=logger,
        slot_extraction_service=slot_extraction_service,
        state_update_service=StateUpdateService(
            pricing_flow_state_service=pricing_flow_state_service,
        ),
        telemetry_service=TelemetryService(logger),
        product_resolver=product_resolver,
        dialog_act_router=DialogActRouter(),
    )


def create_app(config_path: str = DEFAULT_CONFIG_PATH):
    from src.api.factory import ApiAppFactory

    project_root = Path(__file__).resolve().parent.parent
    chatbot = build_app(project_root=project_root, config_path=config_path)
    factory = ApiAppFactory(
        chatbot=chatbot,
        api_key_env_var="CHATBOT_API_KEY",
        api_key_header_name="X-API-Key",
    )
    return factory.create_app()


def run_cli() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app = build_app(project_root)

    print("Autopoisk Knowledge Bot запущен. Напишите вопрос.")
    print("Команды: /exit, /quit, /clear")

    session_id = "terminal-session"
    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("Завершение работы.")
            break

        if user_input.lower() == "/clear":
            print("История сессии очищена.")
            app.clear_session(session_id)
            continue

        response = app.respond(session_id=session_id, user_query=user_input)
        print(f"\nБот: {response.answer_text}")

        if response.media_refs:
            print("\nМедиа:")
            for media in response.media_refs:
                print(f"- {media.media_type}: {media.url} ({media.description})")


def run_api(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    import uvicorn

    project_root = Path(__file__).resolve().parent.parent
    config = ConfigLoader(project_root / config_path).load()
    app = create_app(config_path=config_path)
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


def _missing_dependency_asgi_app(message: str):
    async def _app(scope, receive, send):
        if scope.get("type") != "http":
            return
        body = message.encode("utf-8")
        headers = [(b"content-type", b"text/plain; charset=utf-8")]
        await send({"type": "http.response.start", "status": 500, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return _app


def _build_module_app():
    try:
        return create_app()
    except ModuleNotFoundError as error:
        missing = error.name or "required dependency"
        return _missing_dependency_asgi_app(
            "API dependencies are not installed. "
            f"Missing module: {missing}. "
            "Install requirements: /root/project/.venv/bin/pip install -r /root/project/Chat_bot/requirements.txt"
        )
    except RuntimeError as error:
        return _missing_dependency_asgi_app(str(error))


try:
    app = _build_module_app()
except Exception as error:
    app = _missing_dependency_asgi_app(f"API app bootstrap failed: {error}")


if __name__ == "__main__":
    run_cli()

# cd /root/project/Chat_bot/src && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8003 --reload
# http://170.168.12.24:8009/docs

# Очистить лог файл
# truncate -s 0 /root/project/Chat_bot/logs/chatbot.log
# truncate -s 0 /root/project/Chat_bot/logs/tests.log
# cd /root/project/Chat_bot && CHATBOT_API_KEY="xK9mLpQ2vN7wR" /root/project/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8009 --reload --reload-dir /root/project/Chat_bot
