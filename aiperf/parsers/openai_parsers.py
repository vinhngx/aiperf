# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import orjson

from aiperf.common.enums import EndpointType, OpenAIObjectType
from aiperf.common.factories import (
    FactoryCreationError,
    OpenAIObjectParserFactory,
    ResponseExtractorFactory,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    BaseResponseData,
    EmbeddingResponseData,
    InferenceServerResponse,
    ParsedResponse,
    RankingsResponseData,
    ReasoningResponseData,
    RequestRecord,
    SSEMessage,
    TextResponse,
    TextResponseData,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.protocols import OpenAIObjectParserProtocol
from aiperf.common.utils import load_json_str


@ResponseExtractorFactory.register_all(
    EndpointType.CHAT,
    EndpointType.COMPLETIONS,
    EndpointType.EMBEDDINGS,
    EndpointType.RANKINGS,
)
class OpenAIResponseExtractor(AIPerfLoggerMixin):
    """Extractor for OpenAI responses."""

    def __init__(self, model_endpoint: ModelEndpointInfo) -> None:
        """Create a new response extractor based on the provided configuration."""
        super().__init__()
        self.model_endpoint = model_endpoint

    async def extract_response_data(
        self, record: RequestRecord
    ) -> list[ParsedResponse]:
        """Extract the text from a server response message."""
        results = []
        for response in record.responses:
            response_data = self._parse_response(response)
            if not response_data:
                continue
            results.append(response_data)
        return results

    def _parse_response(
        self, response: InferenceServerResponse
    ) -> ParsedResponse | None:
        """Parse a response into a ParsedResponse object."""
        parsed_data = None
        # Note, this uses Python 3.10+ pattern matching, no new objects are created
        match response:
            case TextResponse():
                parsed_data = self._parse_raw_text(response.text)
            case SSEMessage():
                parsed_data = self._parse_raw_text(response.extract_data_content())
            case _:
                self.warning(f"Unsupported response type: {type(response)}")
        if not parsed_data:
            return None

        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=parsed_data,
        )

    def _parse_raw_text(self, raw_text: str) -> BaseResponseData | None:
        """Parse the raw text of the response using the appropriate parser from OpenAIObjectParserFactory.

        Returns:
            ParsedResponse | None: The parsed response, or None if the response is not a valid or supported OpenAI object.
        """
        if raw_text in ("", None, "[DONE]"):
            return None

        try:
            json_str = load_json_str(raw_text)
        except orjson.JSONDecodeError as e:
            self.warning(f"Invalid JSON: {raw_text} - {e!r}")
            return None

        if "object" in json_str:
            try:
                object_type = OpenAIObjectType(json_str["object"])
            except ValueError:
                self.warning(
                    f"Unsupported OpenAI object type received: {json_str['object']}"
                )
                return None
        else:
            object_type = self._infer_object_type(json_str)
            if object_type is None:
                return None

        try:
            parser = OpenAIObjectParserFactory.get_or_create_instance(object_type)
            return parser.parse(json_str)
        except FactoryCreationError:
            self.warning(f"No parser found for object type: {object_type!r}")
            return None

    def _infer_object_type(self, json_obj: dict[str, Any]) -> OpenAIObjectType | None:
        """Infer the object type from the JSON structure for responses without explicit 'object' field."""
        if "rankings" in json_obj:
            return OpenAIObjectType.RANKINGS

        self.warning(f"Could not infer object type from response: {json_obj}")
        return None


def _parse_chat_response(
    obj: dict[str, Any], data_key: Literal["message", "delta"]
) -> BaseResponseData | None:
    """Parse the common ChatCompletion and ChatCompletionChunk objects into a ResponseData object.

    Args:
        obj: The JSON object to parse.
        data_key: The key to use to get the data from the choices (e.g. "message" or "delta").

    Returns:
        The parsed response data.
    """
    choices = obj.get("choices")
    if not choices:
        return None

    message = choices[0].get(data_key)
    if not message:
        return None

    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if not content and not reasoning:
        return None

    if not reasoning:
        return _make_text_response_data(content)

    return ReasoningResponseData(
        content=content,
        reasoning=reasoning,
    )


@OpenAIObjectParserFactory.register(OpenAIObjectType.CHAT_COMPLETION)
class ChatCompletionParser(OpenAIObjectParserProtocol):
    """Parser for ChatCompletion objects."""

    def parse(self, obj: dict[str, Any]) -> BaseResponseData | None:
        """Parse a ChatCompletion into a ResponseData object."""
        return _parse_chat_response(obj, "message")


@OpenAIObjectParserFactory.register(OpenAIObjectType.CHAT_COMPLETION_CHUNK)
class ChatCompletionChunkParser(OpenAIObjectParserProtocol):
    """Parser for ChatCompletionChunk objects."""

    def parse(self, obj: dict[str, Any]) -> BaseResponseData | None:
        """Parse a ChatCompletionChunk into a ResponseData object."""
        return _parse_chat_response(obj, "delta")


@OpenAIObjectParserFactory.register_all(
    OpenAIObjectType.COMPLETION, OpenAIObjectType.TEXT_COMPLETION
)
class CompletionParser(OpenAIObjectParserProtocol):
    """Parser for Completion objects."""

    def parse(self, obj: dict[str, Any]) -> BaseResponseData | None:
        """Parse a Completion object."""
        choices = obj.get("choices")
        if not choices:
            return None
        return _make_text_response_data(choices[0].get("text"))


@OpenAIObjectParserFactory.register(OpenAIObjectType.LIST)
class ListParser(OpenAIObjectParserProtocol):
    """Parser for List objects."""

    def parse(self, obj: dict[str, Any]) -> BaseResponseData | None:
        """Parse a List object."""
        data = obj.get("data", [])
        if all(
            isinstance(item, dict) and item.get("object") == OpenAIObjectType.EMBEDDING
            for item in data
        ):
            return _make_embedding_response_data(data)
        else:
            raise ValueError(f"Received invalid list in response: {obj}")


@OpenAIObjectParserFactory.register(OpenAIObjectType.RANKINGS)
class RankingsParser(OpenAIObjectParserProtocol):
    """Parser for Rankings objects."""

    def parse(self, obj: dict[str, Any]) -> BaseResponseData | None:
        """Parse a Rankings object."""
        rankings = obj.get("rankings", [])
        if not rankings:
            return None
        return RankingsResponseData(rankings=rankings)


def _make_text_response_data(text: str | None) -> TextResponseData | None:
    """Make a TextResponseData object from a string or return None if the text is empty."""
    return TextResponseData(text=text) if text else None


def _make_embedding_response_data(
    data: list[dict[str, Any]],
) -> EmbeddingResponseData | None:
    """Make an EmbeddingResponseData object from a list of embedding dictionaries."""
    if not data:
        return None

    embeddings = []
    for item in data:
        embedding = item.get("embedding", [])
        if embedding:
            embeddings.append(embedding)

    return EmbeddingResponseData(embeddings=embeddings) if embeddings else None
