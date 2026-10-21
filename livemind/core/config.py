"""Global configuration management for LiveMind using pydantic-settings."""

from typing import List, Optional, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Settings
    HOST: str = Field(default="0.0.0.0", description="FastAPI host")
    PORT: int = Field(default=8000, description="FastAPI port")
    ENVIRONMENT: str = Field(default="development", description="Environment mode: development, test, production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Jira Connector Settings
    JIRA_BASE_URL: str = Field(default="https://your-domain.atlassian.net")
    JIRA_API_TOKEN: Optional[str] = Field(default=None)
    JIRA_USER_EMAIL: Optional[str] = Field(default=None)
    JIRA_PROJECT_KEYS: str = Field(
        default="PAY,CORE,INFRA",
        description="Comma-separated list of recognized Jira project keys",
    )

    # Notion Connector Settings
    NOTION_API_KEY: Optional[str] = Field(default=None)
    NOTION_DATABASE_ID: Optional[str] = Field(default=None)

    # Retrieval & Search Settings
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    HYBRID_RRF_K: int = Field(default=60)
    HYBRID_RETRIEVAL_TOP_K: int = Field(default=5)

    # Memory Service Settings
    MEMORY_STORAGE_PATH: str = Field(default="./livemind_memory.db")
    MEMORY_DEFAULT_TTL_DAYS: int = Field(default=30)
    MEMORY_SIMILARITY_THRESHOLD: float = Field(default=0.75)

    # Multi-Hop Reasoning Agent Settings
    RESEARCH_MAX_HOPS: int = Field(default=4)
    RESEARCH_CONFIDENCE_THRESHOLD: float = Field(default=0.70)

    # Security & API Auth
    REQUIRE_API_KEY: bool = Field(default=False)
    API_KEY: str = Field(default="livemind-default-secret-key-v1")

    @property
    def recognized_project_keys(self) -> Set[str]:
        """Returns set of uppercase project keys configured for Jira."""
        return {k.strip().upper() for k in self.JIRA_PROJECT_KEYS.split(",") if k.strip()}


# Global singleton instance
settings = Settings()
