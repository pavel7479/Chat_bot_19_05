# the_First_Agent

Изолированный контур первого агента.

Сюда переносится только то, что относится к цепочке работы первого агента:
- запрос + история + предыдущее состояние
- нормализация ввода
- переформулирование вопроса без добавления новых фактов
- формирование prompt для модели
- разбор JSON-ответа модели
- trace/diagnostics первого агента

Папки:
- `orchestrator/` — основной orchestration-код первого агента
- `catalog/` — работа со списком intent/тем первого агента
- `preprocessing/` — нормализация и rewrite
- `parsing/` — разбор model JSON
- `context/` — prompt/context/history helpers
- `prompts/` — prompt-файлы первого агента
- `config/` — config первого агента

Не переносится внутрь этого пакета то, что является shared для всего проекта:
- общие data-модели
- общий LLM provider
- общий config loader
- общий telemetry/logger
