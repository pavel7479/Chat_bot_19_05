from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import ChatRequest, ChatResponse, HealthResponse
from src.api.security import ApiKeyValidator
from src.app.chatbot import ChatBotOrchestrator


@dataclass(slots=True)
class ApiAppFactory:
    chatbot: ChatBotOrchestrator
    api_key_env_var: str = "CHATBOT_API_KEY"
    api_key_header_name: str = "X-API-Key"

    def create_app(self) -> FastAPI:
        app = FastAPI(title="Autopoisk ChatBot API")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        key_validator = ApiKeyValidator.from_env(
            env_var_name=self.api_key_env_var,
            header_name=self.api_key_header_name,
        )
        require_api_key = key_validator.dependency()

        @app.get("/health", response_model=HealthResponse)
        def health() -> HealthResponse:
            return HealthResponse()

        @app.post(
            "/chat",
            response_model=ChatResponse,
            dependencies=[Depends(require_api_key)],
        )
        def chat(request: ChatRequest) -> ChatResponse:
            response = self.chatbot.respond(session_id=request.client_id, user_query=request.message)
            return ChatResponse(answer=response.answer_text)

        return app
