"""Direct Gemini provider using the Google Gen AI Python SDK."""

import json
import os
from typing import Any, Optional

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "Gemini provider requires the 'google-genai' package. "
        "Install it with: pip install 'aisuite[gemini]' or pip install google-genai"
    ) from e

from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import (
    ChatCompletionMessageToolCall,
    CompletionUsage,
    Function,
    Message,
)
from aisuite.provider import LLMError, Provider


DEFAULT_TEMPERATURE = 0.7


class GeminiMessageConverter:
    """Convert between aisuite's OpenAI-like schema and Gemini's content schema."""

    @staticmethod
    def _supports_function_declaration_field(field_name: str) -> bool:
        model_fields = getattr(types.FunctionDeclaration, "model_fields", None)
        if model_fields is None:
            return False
        return field_name in model_fields

    @staticmethod
    def _message_to_dict(message: Any) -> dict:
        return message.model_dump(mode="json") if hasattr(message, "model_dump") else message

    @staticmethod
    def _is_native_gemini_tool(tool: Any) -> bool:
        return isinstance(tool, types.Tool)

    def convert_request(self, messages: list) -> tuple[Optional[str], list[dict]]:
        system_messages = []
        converted_messages = []

        for message in messages:
            message_dict = self._message_to_dict(message)
            role = message_dict["role"]

            if role == "system":
                if message_dict.get("content"):
                    system_messages.append(message_dict["content"])
                continue

            converted_messages.append(self._convert_single_message(message_dict))

        system_instruction = "\n\n".join(system_messages) if system_messages else None
        return system_instruction, converted_messages

    def _convert_single_message(self, message: dict) -> dict:
        role = message["role"]

        if role == "user":
            return {"role": "user", "parts": [{"text": message["content"]}]}

        if role == "assistant":
            parts = []
            if message.get("content"):
                parts.append({"text": message["content"]})

            for tool_call in message.get("tool_calls") or []:
                tool_name, arguments, _ = self._extract_tool_call_fields(tool_call)
                parts.append(
                    {
                        "function_call": {
                            "name": tool_name,
                            "args": arguments,
                        }
                    }
                )

            return {"role": "model", "parts": parts}

        if role == "tool":
            content = message.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"output": content}

            function_response = {
                "name": message["name"],
                "response": content,
            }

            if message.get("tool_call_id"):
                function_response["id"] = message["tool_call_id"]

            return {
                "role": "tool",
                "parts": [{"function_response": function_response}],
            }

        raise ValueError(f"Unsupported message role for Gemini: {role}")

    def convert_response(self, response: Any) -> ChatCompletionResponse:
        normalized_response = ChatCompletionResponse()

        text_content = self._extract_text(response)
        function_calls = getattr(response, "function_calls", None) or []

        tool_calls = None
        if function_calls:
            tool_calls = []
            for index, function_call in enumerate(function_calls):
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=getattr(function_call, "id", None)
                        or f"call_{index}_{function_call.name}",
                        type="function",
                        function=Function(
                            name=function_call.name,
                            arguments=json.dumps(function_call.args or {}),
                        ),
                    )
                )

        normalized_response.choices[0].message = Message(
            content=text_content,
            role="assistant",
            tool_calls=tool_calls,
            refusal=None,
        )
        normalized_response.choices[0].finish_reason = (
            "tool_calls" if tool_calls else "stop"
        )

        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata:
            normalized_response.usage = CompletionUsage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", None),
                completion_tokens=getattr(
                    usage_metadata, "candidates_token_count", None
                ),
                total_tokens=getattr(usage_metadata, "total_token_count", None),
            )

        return normalized_response

    def _extract_text(self, response: Any) -> Optional[str]:
        try:
            text = response.text
            if text:
                return text
        except Exception:
            pass

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        text_parts = [part.text for part in parts if getattr(part, "text", None)]
        return "".join(text_parts) if text_parts else None

    @staticmethod
    def _extract_tool_call_fields(tool_call: Any) -> tuple[str, dict, Optional[str]]:
        if isinstance(tool_call, dict):
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return (
                tool_call["function"]["name"],
                arguments,
                tool_call.get("id"),
            )

        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return tool_call.function.name, arguments, tool_call.id

    def convert_tools(self, tools: list[Any]) -> tuple[list[types.Tool], bool]:
        converted_tools = []
        has_function_tools = False

        for tool in tools:
            if self._is_native_gemini_tool(tool):
                converted_tools.append(tool)
                continue

            if not isinstance(tool, dict):
                continue

            if tool.get("type") != "function":
                continue

            function = tool["function"]
            function_declaration_kwargs = {
                "name": function["name"],
                "description": function.get("description", ""),
            }

            parameters = function.get("parameters", {})
            if self._supports_function_declaration_field("parameters_json_schema"):
                function_declaration_kwargs["parameters_json_schema"] = parameters
            else:
                function_declaration_kwargs["parameters"] = parameters
            has_function_tools = True

            converted_tools.append(
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**function_declaration_kwargs)
                    ]
                )
            )
        return converted_tools, has_function_tools

    def prepare_config(
        self, kwargs: dict, system_instruction: Optional[str]
    ) -> types.GenerateContentConfig:
        config = kwargs.copy()

        config.setdefault("temperature", DEFAULT_TEMPERATURE)

        if "max_tokens" in config:
            config["max_output_tokens"] = config.pop("max_tokens")

        if "stop" in config:
            stop_sequences = config.pop("stop")
            config["stop_sequences"] = (
                stop_sequences
                if isinstance(stop_sequences, list)
                else [stop_sequences]
            )

        if system_instruction:
            config["system_instruction"] = system_instruction

        if "tools" in config:
            converted_tools, _ = self.convert_tools(config["tools"])
            config["tools"] = converted_tools

        tool_choice = config.pop("tool_choice", None)
        if tool_choice is not None:
            function_calling_config = None

            if tool_choice == "none":
                function_calling_config = types.FunctionCallingConfig(mode="NONE")
            elif tool_choice == "required":
                function_calling_config = types.FunctionCallingConfig(mode="ANY")
            elif isinstance(tool_choice, dict):
                function_name = tool_choice.get("function", {}).get("name")
                if function_name:
                    function_calling_config = types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=[function_name],
                    )

            if function_calling_config:
                config["tool_config"] = types.ToolConfig(
                    function_calling_config=function_calling_config
                )

        return types.GenerateContentConfig(**config)


class GeminiProvider(Provider):
    """Direct Gemini provider via googleapis/python-genai."""

    def __init__(self, **config):
        api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_API_KEY"
        )
        if not api_key:
            raise ValueError(
                "Gemini API key is missing. Please provide it in the config or set "
                "GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )

        config = self._normalize_client_config(config)
        config.setdefault("api_key", api_key)
        self.client = genai.Client(**config)
        self.transformer = GeminiMessageConverter()

    def _normalize_client_config(self, config: dict) -> dict:
        """Map common transport aliases to google-genai's http_options structure."""
        normalized = dict(config)
        http_options = dict(normalized.pop("http_options", {}) or {})

        base_url = normalized.pop("base_url", None)
        if base_url is not None:
            http_options["base_url"] = base_url

        headers = normalized.pop("headers", None)
        extra_headers = normalized.pop("extra_headers", None)
        merged_headers = dict(http_options.get("headers", {}) or {})
        if headers:
            merged_headers.update(headers)
        if extra_headers:
            merged_headers.update(extra_headers)
        if merged_headers:
            http_options["headers"] = merged_headers

        timeout = normalized.pop("timeout", None)
        if timeout is not None:
            http_options["timeout"] = timeout

        if http_options:
            normalized["http_options"] = http_options

        return normalized

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            system_instruction, contents = self.transformer.convert_request(messages)
            config = self.transformer.prepare_config(kwargs, system_instruction)
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return self.transformer.convert_response(response)
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
