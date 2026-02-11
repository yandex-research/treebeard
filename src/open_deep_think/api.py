"""API client for single-turn chat completions (OpenAI-compatible)."""

import os

from openai import OpenAI
from openai.types.chat import ChatCompletion

# Default timeout for API requests (seconds)
_REQUEST_TIMEOUT = 1800


def single_turn_api_call(model: str, prompt: str, max_tokens: int) -> ChatCompletion:
    """Call an OpenAI-compatible (OpenAPI) chat completions API with the given prompt.

    Uses the OpenAI SDK. Expects API_KEY and optionally API_BASE_URL
    (e.g. https://api.openai.com/v1) in the environment.

    Args:
        model: Model identifier (e.g., "gpt-4o")
        prompt: The problem text to send as the user message
        max_tokens: Maximum tokens for the response

    Returns:
        OpenAI SDK ChatCompletion (e.g. .choices[0].message.content for the reply).

    """
    api_key = os.environ.get("API_KEY")
    if not api_key:
        msg = "API key must be provided or set in API_KEY environment variable"
        raise ValueError(msg)

    base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(_REQUEST_TIMEOUT),
    )

    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
