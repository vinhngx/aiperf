# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.enums import EndpointType
from aiperf.common.factories import RequestConverterFactory
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo

DEFAULT_ROLE = "user"


@RequestConverterFactory.register(EndpointType.CHAT)
class OpenAIChatCompletionRequestConverter(AIPerfLoggerMixin):
    """Request converter for OpenAI chat completion requests."""

    async def format_payload(
        self,
        model_endpoint: ModelEndpointInfo,
        turns: list[Turn],
    ) -> dict[str, Any]:
        """Format payload for a chat completion request."""
        messages = self._create_messages(turns)

        payload = {
            "messages": messages,
            "model": turns[-1].model or model_endpoint.primary_model_name,
            "stream": model_endpoint.endpoint.streaming,
        }

        if turns[-1].max_tokens is not None:
            payload["max_completion_tokens"] = turns[-1].max_tokens

        if model_endpoint.endpoint.extra:
            payload.update(model_endpoint.endpoint.extra)

        self.debug(lambda: f"Formatted payload: {payload}")
        return payload

    def _create_messages(self, turns: list[Turn]) -> list[dict[str, Any]]:
        messages = []
        for turn in turns:
            message = {
                "role": turn.role or DEFAULT_ROLE,
            }
            if (
                len(turn.texts) == 1
                and len(turn.texts[0].contents) == 1
                and len(turn.images) == 0
                and len(turn.audios) == 0
                and len(turn.videos) == 0
            ):
                # Hotfix for Dynamo API which does not yet support a list of messages
                message["name"] = turn.texts[0].name
                message["content"] = (
                    turn.texts[0].contents[0] if turn.texts[0].contents else ""
                )
                messages.append(message)
                continue
            message_content: list[dict[str, Any]] = []

            for text in turn.texts:
                for content in text.contents:
                    if not content:
                        continue
                    message_content.append({"type": "text", "text": content})

            for image in turn.images:
                for content in image.contents:
                    if not content:
                        continue
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": content}}
                    )

            for audio in turn.audios:
                for content in audio.contents:
                    if not content:
                        continue
                    if "," not in content:
                        raise ValueError(
                            "Audio content must be in the format 'format,b64_audio'."
                        )
                    format, b64_audio = content.split(",", 1)
                    message_content.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": b64_audio,
                                "format": format,
                            },
                        }
                    )
            for video in turn.videos:
                for content in video.contents:
                    if not content:
                        continue
                    message_content.append(
                        {"type": "video_url", "video_url": {"url": content}}
                    )
            message["content"] = message_content
            messages.append(message)
        return messages
