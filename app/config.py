import os
from loguru import logger


class BaseConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    JSON_SORT_KEYS = False
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024


class DevelopmentConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class TestingConfig(BaseConfig):
    TESTING = True


def get_config(name: str | None):
    env = (name or os.getenv("FASTAPI_ENV", "production")).lower()
    return {"development": DevelopmentConfig, "testing": TestingConfig}.get(env, ProductionConfig)


# Module import log
logger.debug(f"[{__name__}] module loaded")


