"""API key authentication."""
from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from .config import API_KEY

_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_header)) -> str | None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")
    return key
