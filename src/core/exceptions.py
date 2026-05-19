class ChatBotError(Exception):
    """Base chatbot exception."""


class ConfigurationError(ChatBotError):
    """Raised when config is invalid."""


class LLMProviderError(ChatBotError):
    """Raised when LLM call fails."""
