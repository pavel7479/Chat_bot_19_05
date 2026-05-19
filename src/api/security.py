from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader


@dataclass(slots=True)
class ApiKeyValidator:
    expected_api_key: str
    header_name: str = "X-API-Key"

    @classmethod
    def from_env(cls, env_var_name: str = "CHATBOT_API_KEY", header_name: str = "X-API-Key") -> "ApiKeyValidator":
        api_key = os.getenv(env_var_name)
        if not api_key:
            raise RuntimeError(
                f"{env_var_name} environment variable is not set. "
                "Start API with CHATBOT_API_KEY='<secret>' ..."
            )
        return cls(expected_api_key=api_key, header_name=header_name)

    def dependency(self):
        scheme = APIKeyHeader(name=self.header_name, auto_error=False)

        async def require_api_key(provided_key: str | None = Security(scheme)) -> None:
            if provided_key != self.expected_api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized",
                )

        return require_api_key
