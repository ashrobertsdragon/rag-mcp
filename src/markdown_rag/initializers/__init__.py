"""Initialize embeddings model."""

from langchain_core.embeddings import Embeddings

from markdown_rag.config import Env, GoogleEnv, OllamaEnv
from markdown_rag.models import EmbeddingEngine


def initialize_google(settings: GoogleEnv) -> Embeddings:
    """Initialize Google embeddings."""
    from markdown_rag.initializers.google import initialize

    return initialize(settings)


def initialize_ollama(settings: OllamaEnv) -> Embeddings:
    """Initialize Ollama embeddings."""
    from markdown_rag.initializers.ollama import initialize

    return initialize(settings)


def initialize_embeddings(
    embeddings_engine: EmbeddingEngine, settings: Env
) -> Embeddings:
    """Initialize the embeddings model."""
    engine = {
        EmbeddingEngine.OLLAMA: initialize_ollama,
        EmbeddingEngine.GOOGLE: initialize_google,
    }
    return engine[embeddings_engine](settings)
