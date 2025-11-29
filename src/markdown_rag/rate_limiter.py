"""Rate limiter for API requests."""

import logging
import time
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import lru_cache

from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.ai.generativelanguage_v1beta.types import Content, Part

logger = logging.getLogger(__name__)


class TokenCounter:
    """Functor for counting tokens in a prompt."""

    def __init__(self, client: GenerativeServiceClient, model: str):
        """Initialize the token counter.

        Args:
            client: The generative AI client.
            model: The model name to use for tokenization.
        """
        self.client = client
        self.model = model

    def __call__(self, prompt: str) -> int:
        """Count tokens in a prompt.

        Args:
            prompt: The prompt string.

        Returns:
            The number of tokens in the prompt.
        """
        contents = [Content(parts=[Part(text=prompt)])]
        response = self.client.count_tokens(
            model=self.model, contents=contents
        )
        return response.total_tokens


@dataclass
class Request:
    """Single embedding request record metadata."""

    timestamp: float
    token_count: int


@dataclass
class UsageStats:
    """Current API usage statistics."""

    minute_requests: int
    day_requests: int
    minute_tokens: int


class UsageTracker:
    """Efficiently tracks API usage with cached statistics.

    Maintains a sliding window of requests and provides O(1) amortized
    access to current usage stats via caching.
    """

    def __init__(self, minute_window: float, day_window: float):
        """Initialize the usage tracker.

        Args:
            minute_window: Time window in seconds for minute-based limits
            day_window: Time window in seconds for day-based limits
        """
        self._requests: deque[Request] = deque()
        self._minute_window = minute_window
        self._day_window = day_window
        self._cached_stats: UsageStats | None = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 0.1

    def add_request(self, timestamp: float, tokens: int) -> None:
        """Record a new request."""
        self._requests.append(Request(timestamp, tokens))
        self._invalidate_cache()

    def cleanup_old(self, current_time: float) -> None:
        """Remove requests outside the daily window."""
        cutoff = current_time - self._day_window
        while self._requests and self._requests[0].timestamp < cutoff:
            self._requests.popleft()
        self._invalidate_cache()

    def get_stats(self, current_time: float) -> UsageStats:
        """Get current usage statistics with caching."""
        if (
            self._cached_stats
            and (current_time - self._cache_time) < self._cache_ttl
        ):
            return self._cached_stats

        self._cached_stats = self._calculate_stats(current_time)
        self._cache_time = current_time
        return self._cached_stats

    def find_oldest_in_window(
        self, current_time: float, window: float
    ) -> float | None:
        """Find the oldest request timestamp within a window."""
        cutoff = current_time - window
        for req in self._requests:
            if req.timestamp > cutoff:
                return req.timestamp
        return None

    def _calculate_stats(self, current_time: float) -> UsageStats:
        """Calculate usage stats by scanning the deque once."""
        minute_cutoff = current_time - self._minute_window
        day_cutoff = current_time - self._day_window

        minute_requests = 0
        day_requests = 0
        minute_tokens = 0

        for req in reversed(self._requests):
            if req.timestamp >= minute_cutoff:
                minute_requests += 1
                minute_tokens += req.token_count
            if req.timestamp >= day_cutoff:
                day_requests += 1
            else:
                break

        return UsageStats(minute_requests, day_requests, minute_tokens)

    def _invalidate_cache(self) -> None:
        """Invalidate the cached statistics."""
        self._cached_stats = None


@lru_cache(maxsize=20)
def get_token_count(prompt: str, tokenizer: Callable[[str], int]) -> int:
    """Request token count from tokenizer.

    Cached for recursive calls.

    Args:
        prompt: Prompt to count tokens for

    Returns:
        int: Token count
    """
    return tokenizer(prompt)


class RateLimiter:
    """Rate limiter using sliding window algorithm for API request tracking.

    Tracks requests and tokens within specified time windows to enforce limits.
    """

    minute_window: float = 60.0
    day_window: float = 86400.0

    def __init__(
        self,
        tokenizer: Callable[[str], int],
        max_requests_per_minute: int = 100,
        max_tokens_per_minute: int = 30000,
        max_requests_per_day: int = 1000,
    ):
        """Constructor for RateLimiter.

        Args:
            tokenizer: Tokenizer function
            max_requests_per_minute: Maximum requests per minute
            max_tokens_per_minute: Maximum tokens per minute
            max_requests_per_day: Maximum requests per day
        """
        self._tracker = UsageTracker(self.minute_window, self.day_window)
        self._tokenizer = tokenizer
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_day = max_requests_per_day

    def _calculate_wait_time(
        self, current_time: float, tokens: int
    ) -> tuple[float, UsageStats]:
        """Calculate wait time based on current usage and new tokens.

        Args:
            current_time: Current timestamp
            tokens: Number of tokens in the new request

        Returns:
            Tuple of (wait_time, current_usage_stats)
        """
        stats = self._tracker.get_stats(current_time)
        total_tokens = stats.minute_tokens + tokens

        wait_times = []

        if stats.minute_requests >= self.max_requests_per_minute:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.minute_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.minute_window))

        if stats.day_requests >= self.max_requests_per_day:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.day_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.day_window))

        if total_tokens > self.max_tokens_per_minute:
            oldest = self._tracker.find_oldest_in_window(
                current_time, self.minute_window
            )
            if oldest:
                wait_times.append(oldest - (current_time - self.minute_window))

        return max(wait_times, default=0.0), stats

    MIN_LOG_INTERVAL = 60.0
    MAX_LOG_INTERVAL = 1800.0

    def _get_log_interval(self, wait_time: float) -> float:
        """Determine the logging interval based on wait time.

        Args:
            wait_time: Total wait time in seconds.

        Returns:
            The interval in seconds to log progress.
        """
        # Formula: interval is half the wait time, clamped between min and max
        return min(
            max(self.MIN_LOG_INTERVAL, wait_time / 2.0), self.MAX_LOG_INTERVAL
        )

    def _sleep_with_progress(self, wait_time: float, interval: float) -> None:
        """Sleep for the given time, logging progress at intervals.

        Args:
            wait_time: Total time to wait in seconds.
            interval: Interval in seconds to log progress.
        """
        if wait_time <= 0:
            return

        end_time = time.monotonic() + wait_time

        while (remaining := end_time - time.monotonic()) > interval:
            time.sleep(interval)
            # Re-calculate remaining time after sleep for accurate logging
            if (current_remaining := end_time - time.monotonic()) > 0:
                logger.info(f"Still waiting... {current_remaining:.2f}s remaining")

        if (final_remaining := end_time - time.monotonic()) > 0:
            time.sleep(final_remaining)

    def _wait_and_log(
        self, wait_time: float, stats: UsageStats, tokens: int
    ) -> None:
        """Wait for the specified time and log the reason.

        Args:
            wait_time: Time to wait in seconds
            stats: Current usage statistics
            tokens: Number of tokens in the pending request
        """
        logger.info(
            f"Rate limit approaching. Waiting {wait_time:.2f}s "
            f"(requests: {stats.minute_requests}/"
            f"{self.max_requests_per_minute}/min, "
            f"{stats.day_requests}/{self.max_requests_per_day}/day), "
            f"Tokens: {stats.minute_tokens + tokens}/"
            f"{self.max_tokens_per_minute} tokens/min"
        )

        interval = self._get_log_interval(wait_time)
        self._sleep_with_progress(wait_time, interval)

    def _process_single_request(self, prompt: str) -> None:
        """Process a single request, waiting if necessary.

        Args:
            prompt: The text prompt to process
        """
        tokens = get_token_count(prompt, self._tokenizer)
        current_time = time.time()
        self._tracker.cleanup_old(current_time)

        wait_time, stats = self._calculate_wait_time(current_time, tokens)

        if wait_time <= 0:
            self._tracker.add_request(current_time, tokens)
            logger.debug(
                f"Request allowed: {stats.minute_requests + 1} req/min, "
                f"{stats.day_requests + 1} req/day, "
                f"{stats.minute_tokens + tokens} tokens/min"
            )
            return

        self._wait_and_log(wait_time, stats, tokens)
        self._process_single_request(prompt)

    def _calculate_batch_size(
        self, texts: list[str], current_time: float
    ) -> int:
        """Calculate max batch size that fits within current limits.

        Args:
            texts: List of text prompts to batch
            current_time: Current timestamp

        Returns:
            Maximum number of texts that can be processed in current batch
        """
        stats = self._tracker.get_stats(current_time)

        available_minute = self.max_requests_per_minute - stats.minute_requests
        available_day = self.max_requests_per_day - stats.day_requests
        available_tokens = self.max_tokens_per_minute - stats.minute_tokens

        max_requests = min(available_minute, available_day)
        if max_requests < 1:
            return 0

        cumulative_tokens = 0
        for i, text in enumerate(texts):
            if i >= max_requests:
                return i

            tokens = get_token_count(text, self._tokenizer)
            if cumulative_tokens + tokens > available_tokens:
                return max(1, i)
            cumulative_tokens += tokens

        return len(texts)

    def generate_batches(
        self, texts: list[str]
    ) -> Generator[list[str], None, None]:
        """Generate batches of texts that respect rate limits.

        Handles waiting and approval internally. Each yielded batch is
        ready to send to the API.

        Args:
            texts: List of texts to batch

        Yields:
            Batches of texts to be processed without violating rate limits
        """
        while texts:
            current_time = time.time()
            self._tracker.cleanup_old(current_time)

            batch_size = self._calculate_batch_size(texts, current_time)

            if batch_size < 1:
                self.wait_if_needed(texts)
                continue

            batch = texts[:batch_size]
            self.wait_if_needed(batch)
            yield batch

            texts = texts[batch_size:]

    def wait_if_needed(self, request: list[str] | str) -> None:
        """Block until the request can proceed within rate limits.

        Uses a sliding window algorithm to enforce limits per minute, day, and
        token. Processes each request sequentially to ensure rate limits.

        Args:
            request: Prompt or prompts to count tokens for
        """
        requests = request if isinstance(request, list) else [request]
        for prompt in requests:
            self._process_single_request(prompt)
