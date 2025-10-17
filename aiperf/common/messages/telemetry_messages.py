# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, ProcessTelemetryResult, TelemetryRecord
from aiperf.common.types import MessageTypeT


class TelemetryRecordsMessage(BaseServiceMessage):
    """Message from the telemetry data collector to the records manager to notify it
    of the telemetry records for a batch of GPU samples."""

    message_type: MessageTypeT = MessageType.TELEMETRY_RECORDS

    collector_id: str = Field(
        ...,
        description="The ID of the telemetry data collector that collected the records.",
    )
    records: list[TelemetryRecord] = Field(
        ..., description="The telemetry records collected from GPU monitoring"
    )
    error: ErrorDetails | None = Field(
        default=None, description="The error details if telemetry collection failed."
    )

    @property
    def valid(self) -> bool:
        """Whether the telemetry collection was valid."""

        return self.error is None and len(self.records) > 0


class ProcessTelemetryResultMessage(BaseServiceMessage):
    """Message containing processed telemetry results - mirrors ProcessRecordsResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_TELEMETRY_RESULT

    telemetry_result: ProcessTelemetryResult = Field(
        description="The processed telemetry results"
    )


class TelemetryStatusMessage(BaseServiceMessage):
    """Message from TelemetryManager to SystemController indicating telemetry availability."""

    message_type: MessageTypeT = MessageType.TELEMETRY_STATUS

    enabled: bool = Field(
        description="Whether telemetry collection is enabled and will produce results"
    )
    reason: str | None = Field(
        default=None, description="Reason why telemetry is disabled (if enabled=False)"
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of DCGM endpoint URLs in the configured scope for display",
    )
    endpoints_reachable: list[str] = Field(
        default_factory=list,
        description="List of DCGM endpoint URLs that were reachable and will provide data",
    )
