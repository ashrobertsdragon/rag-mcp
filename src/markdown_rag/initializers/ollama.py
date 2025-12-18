from logging import getLogger

from langchain_ollama import OllamaEmbeddings

from markdown_rag.config import OllamaEnv

logger = getLogger(__name__)


def initialize(settings: OllamaEnv) -> OllamaEmbeddings:
    """Initialize Ollama embeddings model connection."""
    logger.debug("Initializing Ollama embeddings model connection")
    return OllamaEmbeddings(
        model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_HOST
    )
