from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path
import os


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Ignore extra env variables
    )

    # Application settings
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # File settings
    MAX_FILE_SIZE: str = "100MB"
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")

    # Google Cloud settings
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: str = ""

    # Database settings
    DATABASE_URL: str = ""

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Frontend URL
    FRONTEND_URL: str = "http://localhost:8501"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)


settings = Settings()
