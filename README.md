# Chat_bot

Production-ready OOP чат-бот с архитектурой Topic Classifier -> Retrieval -> Answer LLM.

## Запуск

```bash
cd Chat_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main
```

## Настройки

Все настройки в `config.yaml`.

## Замена LLM

Замените `llm.provider` и добавьте новый провайдер в `src/llm/provider_factory.py`.

## Замена поиска

Реализуйте новый класс от `KnowledgeBaseSearcher` и подключите в фабрике поисковика.
