import os
from typing import Optional


class Settings:
    # Auth
    auth_required: bool = os.getenv("HM_AUTH_REQUIRED", "false").lower() == "true"
    jwt_secret: str = os.getenv("HM_JWT_SECRET", "dev-secret-change-me")
    jwt_alg: str = os.getenv("HM_JWT_ALG", "HS256")

    # Rate limiting
    rate_limit_per_minute: int = int(os.getenv("HM_RATE_LIMIT_PER_MIN", "60"))

    # Persistence
    db_url: str = os.getenv("HM_DB_URL", "sqlite:///./healthmorph.db")
    persist_enabled: bool = os.getenv("HM_PERSIST_ENABLED", "true").lower() == "true"

    # Media retention
    retain_media: bool = os.getenv("HM_RETAIN_MEDIA", "false").lower() == "true"
    media_ttl_seconds: int = int(os.getenv("HM_MEDIA_TTL", "86400"))


settings = Settings()
