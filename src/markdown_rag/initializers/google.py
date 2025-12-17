from logging import getLogger

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from markdown_rag.config import GoogleEnv
from markdown_rag.embeddings import RateLimitedEmbeddings
from markdown_rag.rate_limiter import RateLimiter, TokenCounter

logger = getLogger(__name__)


def initialize(settings: GoogleEnv) -> RateLimitedEmbeddings:
    """Initialize Google embeddings model connection."""
    logger.debug("Initializing Google embeddings model connection")
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.GOOGLE_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=settings.GOOGLE_API_KEY,
    )
    logger.debug("Initializing the token counter")
    tokenizer = TokenCounter(
        client=base_embeddings.client, model=base_embeddings.model
    )
    logger.debug("Initializing rate limiter")
    rate_limiter = RateLimiter(
        tokenizer=tokenizer,
        max_requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
        max_requests_per_day=settings.RATE_LIMIT_REQUESTS_PER_DAY,
    )

    logger.debug("Wrapping embeddings with rate limiter")
    return RateLimitedEmbeddings(base_embeddings, rate_limiter)
