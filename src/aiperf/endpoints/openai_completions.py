# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.factories import RequestConverterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo


# TODO: Not fully implemented yet.
@RequestConverterFactory.register(EndpointType.COMPLETIONS)
class OpenAICompletionRequestConverter(AIPerfLoggerMixin):
    """Request converter for OpenAI completion requests."""

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turns: list[Turn],
    ) -> dict[str, Any]:
        """Format payload for a completion request."""

        # This converter does not support multi-turn completions. Only the last turn is used.
        turn = turns[-1]
        prompts = [
            content for text in turn.texts for content in text.contents if content
        ]
        extra = model_endpoint.endpoint.extra or []

        payload = {
            "prompt": prompts,
            "model": turn.model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turn.max_tokens:
            payload["max_tokens"] = turn.max_tokens

        if extra:
            payload.update(extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload
