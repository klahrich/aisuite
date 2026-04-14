# Gemini

This fork supports direct Gemini access through the official [`googleapis/python-genai`](https://github.com/googleapis/python-genai) SDK.

## Install

```shell
pip install 'aisuite[gemini]'
```

## Configure

Set either `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

```shell
export GEMINI_API_KEY="your-gemini-api-key"
```

Or pass the API key directly:

```python
import aisuite as ai

client = ai.Client({
    "gemini": {
        "api_key": "your-gemini-api-key",
    }
})
```

## Usage

```python
import aisuite as ai

client = ai.Client({
    "gemini": {
        "api_key": "your-gemini-api-key",
    }
})

response = client.chat.completions.create(
    model="gemini:gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain recursion in one paragraph."},
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

## Tool Calling

Callable Python tools work with the same `aisuite` tool abstraction:

```python
def get_weather(location: str):
    """Get weather for a city."""
    return {"location": location, "forecast": "sunny"}

client = ai.Client({
    "gemini": {
        "api_key": "your-gemini-api-key",
    }
})

response = client.chat.completions.create(
    model="gemini:gemini-2.5-flash",
    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
    tools=[get_weather],
    return_tool_results=True,
)

print(response.choices[0].tool_results)
```

Gemini-native tools are also supported. For Google Search grounding, pass the SDK tool object through directly:

```python
import aisuite as ai
from google.genai import types

client = ai.Client({
    "gemini": {
        "api_key": "your-gemini-api-key",
    }
})

response = client.chat.completions.create(
    model="gemini:gemini-2.5-flash",
    messages=[{"role": "user", "content": "What happened in AI today?"}],
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

print(response.choices[0].message.content)
```

## Notes

- The `gemini` provider is for the Gemini Developer API via `google-genai`.
- If you are calling an OpenAI-compatible local gateway, use the `openai` provider instead of `gemini`.
- If you set `max_tokens`, aisuite maps that to Gemini's `max_output_tokens`, so low values can truncate the response.
- Gemini-native grounding metadata is not yet normalized onto the `aisuite` response object, so today you only get the grounded text response through the standard interface.
