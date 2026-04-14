"""Tests for direct Gemini provider functionality."""

import json
from unittest.mock import MagicMock, patch

import pytest

from aisuite.framework.chat_completion_response import ChatCompletionResponse
from aisuite.providers.gemini_provider import GeminiProvider, types


@pytest.fixture(autouse=True)
def set_gemini_env_var(monkeypatch):
    """Fixture to set the Gemini API key environment variable for tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-api-key")


def test_gemini_provider_text_response():
    """Test that Gemini provider returns a normalized text response."""
    provider = GeminiProvider()

    mock_response = MagicMock()
    mock_response.text = "mocked-text-response-from-model"
    mock_response.function_calls = []
    mock_response.usage_metadata = MagicMock(
        prompt_token_count=10,
        candidates_token_count=20,
        total_token_count=30,
    )

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_response
    ) as mock_generate:
        response = provider.chat_completions_create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.3,
            max_tokens=256,
            stop=["STOP"],
        )

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["contents"] == [{"role": "user", "parts": [{"text": "Hello!"}]}]
        assert isinstance(call_kwargs["config"], types.GenerateContentConfig)
        assert call_kwargs["config"].temperature == 0.3
        assert call_kwargs["config"].max_output_tokens == 256
        assert call_kwargs["config"].stop_sequences == ["STOP"]

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == "mocked-text-response-from-model"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


def test_gemini_provider_function_call_response():
    """Test that Gemini function calls are normalized into tool_calls."""
    provider = GeminiProvider()

    function_call = MagicMock()
    function_call.name = "get_weather"
    function_call.args = {"location": "San Francisco"}
    function_call.id = "gemini-call-1"

    mock_response = MagicMock()
    mock_response.text = None
    mock_response.function_calls = [function_call]
    mock_response.usage_metadata = None

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_response
    ):
        response = provider.chat_completions_create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        )

        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls[0].id == "gemini-call-1"
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
        assert json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        ) == {"location": "San Francisco"}


def test_gemini_provider_accepts_base_url_and_extra_headers():
    """Test that gateway-style transport options are normalized into http_options."""
    with patch("aisuite.providers.gemini_provider.genai.Client") as mock_client:
        GeminiProvider(
            api_key="test-gemini-api-key",
            base_url="http://localhost:8080",
            extra_headers={"x-bf-cache-key": "session-123"},
            timeout=30_000,
        )

    mock_client.assert_called_once_with(
        api_key="test-gemini-api-key",
        http_options={
            "base_url": "http://localhost:8080",
            "headers": {"x-bf-cache-key": "session-123"},
            "timeout": 30_000,
        },
    )


def test_gemini_provider_converts_tool_result_messages():
    """Test that aisuite tool result messages are converted back into Gemini contents."""
    provider = GeminiProvider()
    mock_response = MagicMock()
    mock_response.text = "done"
    mock_response.function_calls = []
    mock_response.usage_metadata = None

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_response
    ) as mock_generate:
        provider.chat_completions_create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "get_weather",
                    "content": '{"forecast": "sunny"}',
                    "tool_call_id": "call_1",
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        )

        contents = mock_generate.call_args.kwargs["contents"]
        assert contents[1]["role"] == "model"
        assert contents[1]["parts"][0]["function_call"]["name"] == "get_weather"
        assert contents[2]["role"] == "tool"
        assert contents[2]["parts"][0]["function_response"]["name"] == "get_weather"
        assert contents[2]["parts"][0]["function_response"]["response"] == {
            "forecast": "sunny"
        }


def test_gemini_provider_tool_choice_none_maps_to_none_mode():
    """Test that tool_choice='none' is translated to Gemini tool config."""
    provider = GeminiProvider()
    mock_response = MagicMock()
    mock_response.text = "done"
    mock_response.function_calls = []
    mock_response.usage_metadata = None

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_response
    ) as mock_generate:
        provider.chat_completions_create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Do nothing",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            tool_choice="none",
        )

        config = mock_generate.call_args.kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.mode == "NONE"


def test_gemini_provider_passes_through_native_google_search_tool():
    """Test that native Gemini tools are passed through without OpenAI conversion."""
    provider = GeminiProvider()
    mock_response = MagicMock()
    mock_response.text = "done"
    mock_response.function_calls = []
    mock_response.usage_metadata = None
    google_search_tool = types.Tool(google_search=types.GoogleSearch())

    with patch.object(
        provider.client.models, "generate_content", return_value=mock_response
    ) as mock_generate:
        provider.chat_completions_create(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "What happened in AI today?"}],
            tools=[google_search_tool],
        )

        config = mock_generate.call_args.kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.tools == [google_search_tool]
        assert config.automatic_function_calling is None
