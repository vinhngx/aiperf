# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommAddress
from aiperf.common.factories import ResponseExtractorFactory
from aiperf.common.hooks import on_init
from aiperf.common.messages import (
    ConversationTurnRequestMessage,
    ConversationTurnResponseMessage,
    ErrorMessage,
)
from aiperf.common.mixins import CommunicationMixin
from aiperf.common.models import (
    ErrorDetails,
    ParsedResponseRecord,
    RequestRecord,
)
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.record_models import ReasoningResponseData
from aiperf.common.protocols import RequestClientProtocol, ResponseExtractorProtocol
from aiperf.common.tokenizer import Tokenizer


# TODO: Should we create non-tokenizer based parsers?
class InferenceResultParser(CommunicationMixin):
    """InferenceResultParser is responsible for parsing the inference results."""

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
        )
        self.conversation_request_client: RequestClientProtocol = (
            self.comms.create_request_client(
                CommAddress.DATASET_MANAGER_PROXY_FRONTEND,
            )
        )
        self.tokenizers: dict[str, Tokenizer] = {}
        self.user_config: UserConfig = user_config
        self.tokenizer_lock: asyncio.Lock = asyncio.Lock()
        self.model_endpoint: ModelEndpointInfo = ModelEndpointInfo.from_user_config(
            user_config
        )
        self.extractor: ResponseExtractorProtocol = (
            ResponseExtractorFactory.create_instance(
                self.model_endpoint.endpoint.type,
                model_endpoint=self.model_endpoint,
            )
        )

    @on_init
    async def _initialize(self) -> None:
        """Initialize inference result parser-specific components."""
        self.debug("Initializing inference result parser")

        self.extractor = ResponseExtractorFactory.create_instance(
            self.model_endpoint.endpoint.type,
            model_endpoint=self.model_endpoint,
        )

    async def configure(self) -> None:
        """Configure the tokenizers."""
        self.info("Configuring tokenizers for inference result parser")
        begin = time.perf_counter()
        async with self.tokenizer_lock:
            self.tokenizers = {
                model.name: Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model.name,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
                for model in self.model_endpoint.models.models
            }
        duration = time.perf_counter() - begin
        tokenizer_info = {
            model: {
                "class": tokenizer._tokenizer.__class__.__name__,
                "name_or_path": getattr(tokenizer._tokenizer, "name_or_path", ""),
            }
            for model, tokenizer in self.tokenizers.items()
        }
        self.info(f"Initialized tokenizers: {tokenizer_info} in {duration:.2f} seconds")

    async def get_tokenizer(self, model: str) -> Tokenizer:
        """Get the tokenizer for a given model or create it if it doesn't exist."""
        async with self.tokenizer_lock:
            if model not in self.tokenizers:
                self.tokenizers[model] = Tokenizer.from_pretrained(
                    self.user_config.tokenizer.name or model,
                    trust_remote_code=self.user_config.tokenizer.trust_remote_code,
                    revision=self.user_config.tokenizer.revision,
                )
            return self.tokenizers[model]

    async def parse_request_record(
        self, request_record: RequestRecord
    ) -> ParsedResponseRecord:
        """Handle an inference results message."""
        self.trace_or_debug(
            lambda: f"Received inference results message: {request_record}",
            lambda: "Received inference results",
        )

        if request_record.has_error:
            # Even for error records, compute input token count if possible
            try:
                input_token_count = await self.compute_input_token_count(request_record)
            except Exception as e:
                self.warning(f"Error computing input token count for error record: {e}")
                input_token_count = None

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                input_token_count=input_token_count,
            )

        elif request_record.valid:
            try:
                record = await self.process_valid_record(request_record)
                self.debug(
                    lambda: f"Received {len(record.request.responses)} responses, input_token_count: {record.input_token_count}, "
                    f"output_token_count: {record.output_token_count}, reasoning_token_count: {record.reasoning_token_count}"
                )
                return record
            except Exception as e:
                # TODO: We should add an ErrorDetails to the response record and not the request record.
                self.exception(f"Error processing valid record: {e}")
                request_record.error = ErrorDetails.from_exception(e)

                try:
                    input_token_count = await self.compute_input_token_count(
                        request_record
                    )
                except Exception:
                    input_token_count = None

                return ParsedResponseRecord(
                    request=request_record,
                    responses=[],
                    input_token_count=input_token_count,
                )
        else:
            self.warning(f"Received invalid inference results: {request_record}")
            # TODO: We should add an ErrorDetails to response record and not the request record.
            request_record.error = ErrorDetails(
                code=None,
                message="Invalid inference results",
                type="InvalidInferenceResults",
            )

            try:
                input_token_count = await self.compute_input_token_count(request_record)
            except Exception:
                input_token_count = None

            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                input_token_count=input_token_count,
            )

    async def process_valid_record(
        self, request_record: RequestRecord
    ) -> ParsedResponseRecord:
        """Process a valid request record."""
        if request_record.model_name is None:
            self.warning(
                lambda: f"Model name is None, unable to process record: {request_record}"
            )
            return ParsedResponseRecord(
                request=request_record,
                responses=[],
                input_token_count=None,
                output_token_count=None,
            )

        resp = await self.extractor.extract_response_data(request_record)
        input_token_count = await self.compute_input_token_count(request_record)

        output_texts: list[str] = []
        reasoning_texts: list[str] = []
        for response in resp:
            if isinstance(response.data, ReasoningResponseData):
                if response.data.reasoning:
                    reasoning_texts.append(response.data.reasoning)
                if response.data.content:
                    output_texts.append(response.data.content)
            else:
                output_texts.append(response.data.get_text())

        tokenizer = await self.get_tokenizer(request_record.model_name)
        output_token_count = (
            len(tokenizer.encode("".join(output_texts))) if output_texts else None
        )
        reasoning_token_count = (
            len(tokenizer.encode("".join(reasoning_texts))) if reasoning_texts else None
        )

        return ParsedResponseRecord(
            request=request_record,
            responses=resp,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            reasoning_token_count=reasoning_token_count,
        )

    async def get_turn(self, request_record: RequestRecord) -> Turn | None:
        """Get the turn for a given request record."""
        if request_record.turn is not None:
            return request_record.turn

        if request_record.conversation_id is None or request_record.turn_index is None:
            self.warning(
                lambda: f"Conversation ID or turn index is None: {request_record.conversation_id=} {request_record.turn_index=}"
            )
            return None

        turn_response: ConversationTurnResponseMessage = (
            await self.conversation_request_client.request(
                ConversationTurnRequestMessage(
                    service_id=self.id,
                    conversation_id=request_record.conversation_id,
                    turn_index=request_record.turn_index,
                )
            )
        )
        if isinstance(turn_response, ErrorMessage):
            self.error(lambda: f"Error getting turn response: {turn_response}")
            return None

        return turn_response.turn

    async def compute_input_token_count(
        self, request_record: RequestRecord
    ) -> int | None:
        """Compute the number of tokens in the input for a given request record."""
        turn = await self.get_turn(request_record)
        if turn is None:
            return None

        tokenizer = await self.get_tokenizer(request_record.model_name)
        input_token_count = 0
        for text in turn.texts:
            input_token_count += len(tokenizer.encode("".join(text.contents)))
        return input_token_count
