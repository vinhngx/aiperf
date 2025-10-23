# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for exporting raw request/response data with per-record metrics."""

import contextlib

import aiofiles

from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import DEFAULT_RAW_RECORD_EXPORT_BATCH_SIZE
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.data_exporter_enums import DataExporterType, ExportLevel
from aiperf.common.enums.post_processor_enums import RecordProcessorType
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.factories import (
    DataExporterFactory,
    EndpointFactory,
    RecordProcessorFactory,
)
from aiperf.common.mixins import AIPerfLoggerMixin, BufferedJSONLWriterMixin
from aiperf.common.models import (
    MetricRecordMetadata,
    ModelEndpointInfo,
    ParsedResponseRecord,
    RawRecordInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import DataExporterProtocol, RecordProcessorProtocol
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


@implements_protocol(RecordProcessorProtocol)
@RecordProcessorFactory.register(RecordProcessorType.RAW_RECORD_WRITER)
class RawRecordWriterProcessor(BufferedJSONLWriterMixin[RawRecordInfo]):
    """Writes raw request/response data with per-record metrics to JSONL files.

    Each RecordProcessor instance writes to its own file to avoid contention
    and enable efficient parallel I/O in distributed setups.

    File format: JSONL (newline-delimited JSON)
    One complete record per line for streaming efficiency.
    """

    def __init__(
        self,
        service_id: str | None,
        user_config: UserConfig,
        **kwargs,
    ):
        self.service_id = service_id or "processor"
        self.user_config = user_config

        if self.user_config.output.export_level != ExportLevel.RAW:
            raise PostProcessorDisabled(
                f"RawRecordWriter processor is disabled for export level {self.user_config.output.export_level}"
            )

        # Construct output file path: raw_records/raw_records_processor_{id}.jsonl
        output_dir = (
            user_config.output.artifact_directory / OutputDefaults.RAW_RECORDS_FOLDER
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Each processor writes to its own file - avoids locking/contention
        # Sanitize service_id for filename (replace special chars)
        safe_id = self.service_id.replace("/", "_").replace(":", "_").replace(" ", "_")
        output_file = output_dir / f"raw_records_{safe_id}.jsonl"

        self._model_endpoint = ModelEndpointInfo.from_user_config(user_config)
        self._endpoint = EndpointFactory.create_instance(
            self._model_endpoint.endpoint.type,
            model_endpoint=self._model_endpoint,
        )

        # Initialize the buffered writer mixin
        super().__init__(
            output_file=output_file,
            batch_size=DEFAULT_RAW_RECORD_EXPORT_BATCH_SIZE,
            service_id=service_id,
            user_config=user_config,
            **kwargs,
        )

        self.info(
            f"RawRecordWriter initialized: {self.output_file} - "
            "FULL request/response data will be exported (files may be large)"
        )

    def _build_export_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> RawRecordInfo:
        """Build the export record for a single record."""

        payload = self._endpoint.format_payload(
            RequestInfo(
                model_endpoint=self._model_endpoint,
                turns=record.request.turns,
            )
        )
        return RawRecordInfo(
            metadata=metadata,
            start_perf_ns=record.request.start_perf_ns,
            payload=payload,
            request_headers=record.request.request_headers,
            response_headers=None,
            status=record.request.status,
            responses=record.request.responses,
            error=record.request.error,
        )

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> None:
        """Process a single record."""
        # Build export record with full parsed record
        record_export = self._build_export_record(record, metadata)

        # Write using the buffered writer mixin (handles batching and flushing)
        await self.buffered_write(record_export)


@implements_protocol(DataExporterProtocol)
@DataExporterFactory.register(DataExporterType.RAW_RECORD_AGGREGATOR)
class RawRecordAggregator(AIPerfLoggerMixin):
    """Aggregator for raw records."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(**kwargs)
        self.exporter_config = exporter_config
        if self.exporter_config.user_config.output.export_level != ExportLevel.RAW:
            raise DataExporterDisabled(
                f"RawRecordAggregator is disabled for export level {self.exporter_config.user_config.output.export_level}"
            )
        self.output_file = (
            exporter_config.user_config.output.profile_export_raw_jsonl_file
        )
        self.output_dir = (
            exporter_config.user_config.output.artifact_directory
            / OutputDefaults.RAW_RECORDS_FOLDER
        )

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="Raw Records",
            file_path=self.output_file,
        )

    async def export(self) -> None:
        """Aggregate the raw records."""
        if self.exporter_config.user_config.output.export_level != ExportLevel.RAW:
            return

        raw_record_files = list(self.output_dir.glob("raw_records_*.jsonl"))
        if not raw_record_files:
            return

        self.output_file.unlink(missing_ok=True)
        self.info(
            f"Aggregating {len(raw_record_files)} raw record files from {self.output_dir} to {self.output_file}"
        )
        record_count = 0
        async with aiofiles.open(self.output_file, "w") as export_file:
            for file in raw_record_files:
                async with aiofiles.open(file) as f:
                    async for line in f:
                        if line.strip():
                            record_count += 1
                            await export_file.write(line)
                file.unlink(missing_ok=True)

        with contextlib.suppress(OSError):
            self.output_dir.rmdir()

        self.info(f"Aggregated {record_count} raw records to {self.output_file}")
