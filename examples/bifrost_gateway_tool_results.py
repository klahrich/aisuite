"""
Example: use aisuite against a Bifrost/OpenAI-compatible local gateway and
return executed tool results directly to the caller.
"""

import os
import random

import aisuite as ai


def rng() -> float:
    """Return a random number between 0 and 1."""
    return random.random()


client = ai.Client(
    {
        "openai": {
            "base_url": os.getenv("OPENAI_BASE_URL", "http://localhost:8080/v1"),
            "extra_headers": {"x-bf-cache-key": "session-123"},
        }
    }
)

# The outer "openai:" selects aisuite's OpenAI-compatible provider.
# The inner "openai/gpt-4o" is the model format expected by Bifrost.
bifrost_model = os.getenv("BIFROST_MODEL", "openai/gpt-4o")

response = client.chat.completions.create(
    model=f"openai:{bifrost_model}",
    messages=[
        {"role": "user", "content": "Generate a random number between 0 and 1."}
    ],
    tools=[rng],
    return_tool_results=True,
)

print(response.choices[0].tool_results)
