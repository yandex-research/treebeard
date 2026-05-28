"""API client for single-turn chat completions (OpenAI-compatible)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

from openai import APIConnectionError, APIStatusError, OpenAI

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

logger = logging.getLogger(__name__)

# timeout for API requests (seconds)
_REQUEST_TIMEOUT = 3600
_MAX_RETRIES = 10
_INITIAL_BACKOFF = 0.5
_BACKOFF_FACTOR = 1.5
# BadRequestError (400) doesn't necessarily mean bad input
# in some APIs it happens when you exceed rate limit too much
# so we want to retry it too
# in cases of actual bad input it will fail after exhausting all retries
_NON_RETRYABLE_STATUS_CODES = {401, 404, 409, 422}


def _build_client() -> OpenAI:
    """Build an OpenAI-compatible client from environment variables.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ValueError: If API_KEY is missing.

    """
    api_key = os.environ.get("API_KEY")
    if not api_key:
        msg = "API key must be provided or set in API_KEY environment variable"
        raise ValueError(msg)

    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(_REQUEST_TIMEOUT),
    )


def chat_api_call(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
) -> ChatCompletion:
    """Call an OpenAI-compatible chat completion API with custom messages.

    Uses the OpenAI SDK. Expects API_KEY and optionally API_BASE_URL
    (e.g. https://api.openai.com/v1) in the environment.

    Args:
        model: Model identifier (e.g., "gpt-4o")
        messages: Chat messages in OpenAI format.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature for response generation.
        top_p: Nucleus sampling parameter for response generation.

    Returns:
        OpenAI SDK ChatCompletion response.

    """
    client = _build_client()
    extra_body = {}

    if "gpt" in model:
        extra_body["reasoning"] = {
            "effort": "xhigh"
        }

    backoff = _INITIAL_BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
            )
        except APIConnectionError:
            if attempt == _MAX_RETRIES:
                raise
            logger.warning("Connection error (attempt %d/%d), retrying in %.1fs…", attempt, _MAX_RETRIES, backoff)
        except APIStatusError as exc:
            if exc.status_code in _NON_RETRYABLE_STATUS_CODES or attempt == _MAX_RETRIES:
                raise
            logger.warning(
                "API error %d (attempt %d/%d), retrying in %.1fs…",
                exc.status_code, attempt, _MAX_RETRIES, backoff,
            )
        time.sleep(backoff)
        backoff *= _BACKOFF_FACTOR


def single_turn_api_call(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None = None,
    top_p: float | None = None,
) -> ChatCompletion:
    """Call an OpenAI-compatible (OpenAPI) chat completions API with the given prompt.

    Uses the OpenAI SDK. Expects API_KEY and optionally API_BASE_URL
    (e.g. https://api.openai.com/v1) in the environment.

    Args:
        model: Model identifier (e.g., "gpt-4o")
        prompt: The problem text to send as the user message
        max_tokens: Maximum tokens for the response
        temperature: Sampling temperature for response generation.
        top_p: Nucleus sampling parameter for response generation.

    Returns:
        OpenAI SDK ChatCompletion (e.g. .choices[0].message.content for the reply).

    """
    return chat_api_call(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
