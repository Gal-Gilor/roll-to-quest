import logging
from functools import lru_cache
from pathlib import Path

import jinja2
from google import genai
from pinecone import Pinecone
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


@lru_cache
class Config(BaseSettings):
    """Project configuration settings."""

    GOOGLE_CLOUD_PROJECT: str
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GOOGLE_GENAI_USE_VERTEXAI: str = True
    GOOGLE_CLOUD_BUCKET: str
    GEMINI_API_KEY: str | None = None
    GENERATION_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    PINECONE_API_KEY: str
    PINECONE_INDEX_HOST: str

    model_config = SettingsConfigDict(env_file=".env")


config = Config()


def create_genai_client(config: BaseSettings) -> genai.Client:
    """Create a Google GenAI client.

    Prioriotize using Vertex AI if GOOGLE_GENAI_USE_VERTEXAI is True.
    If not, use GEMINI_API_KEY if provided. If neither is set, raise an error.

    Args:
        config (Config): Configuration settings.

    Returns:
        genai.Client: Configured GenAI client instance.

    Raises:
        ValueError: If neither GOOGLE_GENAI_USE_VERTEXAI nor GEMINI_API_KEY is set.
    """

    if config.GOOGLE_GENAI_USE_VERTEXAI:
        return genai.Client(
            project=config.GOOGLE_CLOUD_PROJECT,
            location=config.GOOGLE_CLOUD_LOCATION,
            vertexai=config.GOOGLE_GENAI_USE_VERTEXAI,
        )

    if config.GEMINI_API_KEY:
        return genai.Client(api_key=config.GEMINI_API_KEY)

    if not config.GOOGLE_GENAI_USE_VERTEXAI and not config.GEMINI_API_KEY:
        raise ValueError(
            "Either GOOGLE_GENAI_USE_VERTEXAI must be True or GEMINI_API_KEY must be set."
        )

    return genai.Client()


client = create_genai_client(config)


def create_pinecone_client(config: BaseSettings) -> Pinecone:
    """Create a Pinecone client.

    Args:
        config (Config): Configuration settings.

    Returns:
        Pinecone: Configured Pinecone client instance.
    """

    return Pinecone(api_key=config.PINECONE_API_KEY, pool_threads=30)


pinecone_client = create_pinecone_client(config)


@lru_cache
def create_logger(name: str = "Indexers") -> logging.Logger:
    """Create and configure a logger.

    Args:
        name (str): The name of the logger. Defaults to the module name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(asctime)s -> %(message)s",
    )
    logging.getLogger("google.auth").setLevel(logging.ERROR)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    return logging.getLogger(name)


logger = create_logger()


def create_jinja2_environment(
    templates_directory: Path | str = "templates", enable_async: bool = True
) -> jinja2.Environment:
    """Create a Jinja2 environment for rendering templates.

    Args:
        templates_directory (Path | str): The directory containing the Jinja2 templates.
        enable_async (bool): Whether to enable asynchronous rendering. Defaults to True.

    Returns:
        jinja2.Environment: Configured Jinja2 environment instance.
    """
    root_directory = Path(__file__).parent

    if isinstance(templates_directory, str):
        templates_directory = Path(templates_directory)

    templates_directory = root_directory / templates_directory
    if not templates_directory.is_dir():
        raise ValueError(
            f"Templates directory: '{templates_directory}' "
            "does not exist or is not a directory."
        )
    loader = jinja2.FileSystemLoader(templates_directory)

    return jinja2.Environment(loader=loader, enable_async=enable_async)


jinja2_env_async = create_jinja2_environment(
    templates_directory="templates", enable_async=True
)
