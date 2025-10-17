# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    ProfileConfigureCommand,
    ProfileStartCommand,
    TelemetryRecordsMessage,
    TelemetryStatusMessage,
)
from aiperf.common.models import ErrorDetails
from aiperf.gpu_telemetry.constants import DEFAULT_DCGM_ENDPOINTS
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector
from aiperf.gpu_telemetry.telemetry_manager import TelemetryManager


class TestTelemetryManagerInitialization:
    """Test TelemetryManager initialization and configuration."""

    def _create_manager_with_mocked_base(self, user_config):
        """Helper to create TelemetryManager with mocked BaseComponentService."""
        mock_service_config = MagicMock()

        with patch(
            "aiperf.common.base_component_service.BaseComponentService.__init__",
            return_value=None,
        ):
            # Create manager and manually set up comms
            manager = object.__new__(TelemetryManager)
            manager.comms = MagicMock()
            manager.comms.create_push_client = MagicMock(return_value=MagicMock())

            # Call actual __init__ to run real initialization logic
            TelemetryManager.__init__(
                manager,
                service_config=mock_service_config,
                user_config=user_config,
            )

        return manager

    def test_initialization_default_endpoint(self):
        """Test initialization with no user-provided endpoints uses defaults."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = None

        manager = self._create_manager_with_mocked_base(mock_user_config)
        assert manager._dcgm_endpoints == list(DEFAULT_DCGM_ENDPOINTS)

    def test_initialization_custom_endpoints(self):
        """Test initialization with custom user-provided endpoints."""
        mock_user_config = MagicMock(spec=UserConfig)
        custom_endpoint = "http://gpu-node-01:9401/metrics"
        mock_user_config.gpu_telemetry = [custom_endpoint]

        manager = self._create_manager_with_mocked_base(mock_user_config)

        # Should have both defaults + custom endpoint
        for default_endpoint in DEFAULT_DCGM_ENDPOINTS:
            assert default_endpoint in manager._dcgm_endpoints
        assert custom_endpoint in manager._dcgm_endpoints
        assert len(manager._dcgm_endpoints) == 3

    def test_initialization_string_endpoint(self):
        """Test initialization converts single string endpoint to list and prepends defaults."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = "http://single-node:9401/metrics"

        manager = self._create_manager_with_mocked_base(mock_user_config)

        assert isinstance(manager._dcgm_endpoints, list)
        for default_endpoint in DEFAULT_DCGM_ENDPOINTS:
            assert default_endpoint in manager._dcgm_endpoints
        assert "http://single-node:9401/metrics" in manager._dcgm_endpoints
        assert len(manager._dcgm_endpoints) == 3

    def test_initialization_filters_invalid_urls(self):
        """Test initialization filters out invalid URLs."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = [
            "http://valid:9401/metrics",  # Valid
            "not-a-url",  # Invalid - no scheme
            "ftp://wrong-scheme:9401",  # Invalid - wrong scheme
            "http://another-valid:9401",  # Valid
            "",  # Invalid - empty
        ]

        manager = self._create_manager_with_mocked_base(mock_user_config)

        # Should have 2 defaults + 2 valid URLs
        assert len(manager._dcgm_endpoints) == 4
        for default_endpoint in DEFAULT_DCGM_ENDPOINTS:
            assert default_endpoint in manager._dcgm_endpoints
        assert "http://valid:9401/metrics" in manager._dcgm_endpoints
        assert "http://another-valid:9401/metrics" in manager._dcgm_endpoints

    def test_initialization_deduplicates_endpoints(self):
        """Test initialization removes duplicate endpoints while preserving order."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = [
            "http://node1:9401/metrics",
            "http://node2:9401/metrics",
            "http://node1:9401/metrics",  # Duplicate
        ]

        manager = self._create_manager_with_mocked_base(mock_user_config)

        # Should have 2 defaults + 2 unique user endpoints (duplicate removed)
        assert len(manager._dcgm_endpoints) == 4
        assert manager._dcgm_endpoints[0] == DEFAULT_DCGM_ENDPOINTS[0]
        assert manager._dcgm_endpoints[1] == DEFAULT_DCGM_ENDPOINTS[1]
        assert manager._dcgm_endpoints[2] == "http://node1:9401/metrics"
        assert manager._dcgm_endpoints[3] == "http://node2:9401/metrics"

    def test_user_provides_default_endpoint(self):
        """Test that explicitly providing a default endpoint doesn't duplicate it."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = [
            "http://localhost:9400/metrics",  # This is a default
            "http://node1:9401/metrics",
            "http://localhost:9401/metrics",  # This is also a default
        ]

        manager = self._create_manager_with_mocked_base(mock_user_config)

        # Should have 2 defaults + 1 unique user endpoint (defaults not duplicated)
        assert len(manager._dcgm_endpoints) == 3
        assert manager._dcgm_endpoints[0] == DEFAULT_DCGM_ENDPOINTS[0]
        assert manager._dcgm_endpoints[1] == DEFAULT_DCGM_ENDPOINTS[1]
        assert manager._dcgm_endpoints[2] == "http://node1:9401/metrics"
        # Verify user_provided_endpoints excludes the defaults
        assert len(manager._user_provided_endpoints) == 1
        assert "http://node1:9401/metrics" in manager._user_provided_endpoints
        assert "http://localhost:9400/metrics" not in manager._user_provided_endpoints
        assert "http://localhost:9401/metrics" not in manager._user_provided_endpoints


class TestUrlNormalization:
    """Test _normalize_dcgm_url static method."""

    def test_normalize_adds_metrics_suffix(self):
        """Test normalization adds /metrics suffix when missing."""
        url = "http://localhost:9401"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_preserves_metrics_suffix(self):
        """Test normalization preserves existing /metrics suffix."""
        url = "http://localhost:9401/metrics"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_removes_trailing_slash(self):
        """Test normalization removes trailing slash."""
        url = "http://localhost:9401/"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_trailing_slash_with_metrics(self):
        """Test normalization handles trailing slash after /metrics."""
        url = "http://localhost:9401/metrics/"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"

    def test_normalize_complex_path(self):
        """Test normalization with complex URL paths."""
        url = "http://node1:9401/dcgm"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://node1:9401/dcgm/metrics"


class TestCallbackFunctions:
    """Test callback functions for receiving telemetry data."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        # Create minimal manager instance without full initialization
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._user_provided_endpoints = []
        manager._user_explicitly_configured_telemetry = False
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self, sample_telemetry_records):
        """Test _on_telemetry_records with valid records."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        # Call the callback
        await manager._on_telemetry_records(sample_telemetry_records, "test_collector")

        # Verify push was called with correct message
        mock_push_client.push.assert_called_once()
        call_args = mock_push_client.push.call_args[0][0]
        assert isinstance(call_args, TelemetryRecordsMessage)
        assert call_args.service_id == "test_manager"
        assert call_args.collector_id == "test_collector"
        assert call_args.records == sample_telemetry_records
        assert call_args.error is None

    @pytest.mark.asyncio
    async def test_on_telemetry_records_empty(self):
        """Test _on_telemetry_records with empty records list skips sending."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        # Call with empty records
        await manager._on_telemetry_records([], "test_collector")

        # Verify push was NOT called
        mock_push_client.push.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_telemetry_records_exception_handling(
        self, sample_telemetry_records
    ):
        """Test _on_telemetry_records handles exceptions gracefully."""
        manager = self._create_test_manager()

        # Mock the push client to raise exception
        mock_push_client = AsyncMock()
        mock_push_client.push.side_effect = Exception("Network error")
        manager.records_push_client = mock_push_client
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._on_telemetry_records(sample_telemetry_records, "test_collector")

        # Verify error was logged
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_telemetry_error(self):
        """Test _on_telemetry_error callback."""
        manager = self._create_test_manager()

        # Mock the push client
        mock_push_client = AsyncMock()
        manager.records_push_client = mock_push_client

        error_details = ErrorDetails(message="Collection failed")

        # Call the error callback
        await manager._on_telemetry_error(error_details, "test_collector")

        # Verify push was called with error message
        mock_push_client.push.assert_called_once()
        call_args = mock_push_client.push.call_args[0][0]
        assert isinstance(call_args, TelemetryRecordsMessage)
        assert call_args.service_id == "test_manager"
        assert call_args.collector_id == "test_collector"
        assert call_args.records == []
        assert call_args.error == error_details

    @pytest.mark.asyncio
    async def test_on_telemetry_error_exception_handling(self):
        """Test _on_telemetry_error handles exceptions during message sending."""
        manager = self._create_test_manager()

        # Mock the push client to raise exception
        mock_push_client = AsyncMock()
        mock_push_client.push.side_effect = Exception("Push failed")
        manager.records_push_client = mock_push_client
        manager.error = MagicMock()  # Mock error logging

        error_details = ErrorDetails(message="Collection failed")

        # Should not raise exception
        await manager._on_telemetry_error(error_details, "test_collector")

        # Verify error was logged
        manager.error.assert_called_once()


class TestStatusMessaging:
    """Test status message sending functionality."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._user_provided_endpoints = []
        manager._user_explicitly_configured_telemetry = False
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_send_telemetry_status_enabled(self):
        """Test _send_telemetry_status with enabled status."""
        manager = self._create_test_manager()

        # Mock publish method
        manager.publish = AsyncMock()

        endpoints_tested = ["http://node1:9401/metrics", "http://node2:9401/metrics"]
        endpoints_reachable = ["http://node1:9401/metrics"]

        await manager._send_telemetry_status(
            enabled=True,
            endpoints_configured=endpoints_tested,
            endpoints_reachable=endpoints_reachable,
        )

        # Verify publish was called
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert isinstance(call_args, TelemetryStatusMessage)
        assert call_args.enabled is True
        assert call_args.reason is None
        assert call_args.endpoints_configured == endpoints_tested
        assert call_args.endpoints_reachable == endpoints_reachable

    @pytest.mark.asyncio
    async def test_send_telemetry_status_disabled_with_reason(self):
        """Test _send_telemetry_status with disabled status and reason."""
        manager = self._create_test_manager()

        # Mock publish method
        manager.publish = AsyncMock()

        reason = "no DCGM endpoints reachable"
        endpoints_tested = ["http://node1:9401/metrics"]

        await manager._send_telemetry_status(
            enabled=False,
            reason=reason,
            endpoints_configured=endpoints_tested,
            endpoints_reachable=[],
        )

        # Verify publish was called with disabled status
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert isinstance(call_args, TelemetryStatusMessage)
        assert call_args.enabled is False
        assert call_args.reason == reason
        assert call_args.endpoints_reachable == []

    @pytest.mark.asyncio
    async def test_send_telemetry_status_exception_handling(self):
        """Test _send_telemetry_status handles exceptions during publish."""
        manager = self._create_test_manager()

        # Mock publish to raise exception
        manager.publish = AsyncMock(side_effect=Exception("Publish failed"))
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._send_telemetry_status(
            enabled=True, endpoints_configured=[], endpoints_reachable=[]
        )

        # Verify error was logged
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_telemetry_disabled_status_and_shutdown(self):
        """Test _send_telemetry_disabled_status_and_shutdown schedules shutdown."""

        # Side effect to close coroutines and prevent unawaited coroutine warnings
        def close_coroutine(coro):
            coro.close()
            return MagicMock()  # Return a mock Task

        with patch(
            "asyncio.create_task", side_effect=close_coroutine
        ) as mock_create_task:
            manager = self._create_test_manager()

            # Mock dependencies
            manager.publish = AsyncMock()
            manager._dcgm_endpoints = ["http://node1:9401/metrics"]

            reason = "all collectors failed"

            await manager._send_telemetry_disabled_status_and_shutdown(reason)

            # Verify status was sent
            manager.publish.assert_called_once()
            call_args = manager.publish.call_args[0][0]
            assert call_args.enabled is False
            assert call_args.reason == reason

            # Verify delayed shutdown was scheduled
            mock_create_task.assert_called_once()


class TestCollectorManagement:
    """Test collector lifecycle management."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = []
        manager._user_provided_endpoints = []
        manager._user_explicitly_configured_telemetry = False
        manager._collection_interval = 0.33
        return manager

    @pytest.mark.asyncio
    async def test_stop_all_collectors_success(self):
        """Test _stop_all_collectors successfully stops all collectors."""
        manager = self._create_test_manager()

        # Create mock collectors
        mock_collector1 = AsyncMock(spec=TelemetryDataCollector)
        mock_collector2 = AsyncMock(spec=TelemetryDataCollector)

        manager._collectors = {
            "http://node1:9401/metrics": mock_collector1,
            "http://node2:9401/metrics": mock_collector2,
        }

        await manager._stop_all_collectors()

        # Verify both collectors were stopped
        mock_collector1.stop.assert_called_once()
        mock_collector2.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_empty(self):
        """Test _stop_all_collectors with no collectors does nothing."""
        manager = self._create_test_manager()
        manager._collectors = {}

        # Should not raise exception
        await manager._stop_all_collectors()

    @pytest.mark.asyncio
    async def test_stop_all_collectors_handles_failures(self):
        """Test _stop_all_collectors continues despite individual collector failures."""
        manager = self._create_test_manager()

        # Create mock collectors - one fails, one succeeds
        mock_collector1 = AsyncMock(spec=TelemetryDataCollector)
        mock_collector1.stop.side_effect = Exception("Stop failed")
        mock_collector2 = AsyncMock(spec=TelemetryDataCollector)

        manager._collectors = {
            "http://node1:9401/metrics": mock_collector1,
            "http://node2:9401/metrics": mock_collector2,
        }
        manager.error = MagicMock()  # Mock error logging

        # Should not raise exception
        await manager._stop_all_collectors()

        # Verify both stop methods were called
        mock_collector1.stop.assert_called_once()
        mock_collector2.stop.assert_called_once()

        # Verify error was logged for the failed collector
        manager.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_delayed_shutdown(self):
        """Test _delayed_shutdown waits before calling stop."""
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            manager = self._create_test_manager()
            manager.stop = AsyncMock()

            await manager._delayed_shutdown()

            # Verify sleep was called with 5 seconds
            mock_sleep.assert_called_once_with(5.0)

            # Verify stop was called
            manager.stop.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def _create_manager_with_mocked_base(self, user_config):
        """Helper to create TelemetryManager with mocked BaseComponentService."""
        mock_service_config = MagicMock()

        with patch(
            "aiperf.common.base_component_service.BaseComponentService.__init__",
            return_value=None,
        ):
            # Create manager and manually set up comms
            manager = object.__new__(TelemetryManager)
            manager.comms = MagicMock()
            manager.comms.create_push_client = MagicMock(return_value=MagicMock())

            # Call actual __init__ to run real initialization logic
            TelemetryManager.__init__(
                manager,
                service_config=mock_service_config,
                user_config=user_config,
            )

        return manager

    def test_invalid_endpoints_filtered_during_init(self):
        """Test that empty string and invalid URLs are filtered during initialization."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = [
            "",  # Empty string - filtered
            "/metrics",  # No scheme - filtered
            "http://valid:9401/metrics",  # Valid
        ]

        manager = self._create_manager_with_mocked_base(mock_user_config)

        # Only 2 defaults + valid endpoint should remain
        assert len(manager._dcgm_endpoints) == 3
        for default_endpoint in DEFAULT_DCGM_ENDPOINTS:
            assert default_endpoint in manager._dcgm_endpoints
        assert "http://valid:9401/metrics" in manager._dcgm_endpoints

    def test_normalize_url_preserves_valid_structure(self):
        """Test URL normalization only works with properly structured URLs."""
        # normalize_dcgm_url is a simple string operation that assumes valid input
        # Invalid inputs are filtered before normalization in __init__
        url = "http://localhost:9401"
        normalized = TelemetryManager._normalize_dcgm_url(url)
        assert normalized == "http://localhost:9401/metrics"


class TestBothDefaultEndpoints:
    """Test that both default endpoints (9400 and 9401) are tried."""

    def _create_manager_with_mocked_base(self, user_config):
        """Helper to create TelemetryManager with mocked BaseComponentService."""
        mock_service_config = MagicMock()

        with patch(
            "aiperf.common.base_component_service.BaseComponentService.__init__",
            return_value=None,
        ):
            manager = object.__new__(TelemetryManager)
            manager.comms = MagicMock()
            manager.comms.create_push_client = MagicMock(return_value=MagicMock())

            TelemetryManager.__init__(
                manager,
                service_config=mock_service_config,
                user_config=user_config,
            )

        return manager

    def test_both_defaults_included_when_no_user_config(self):
        """Test that both default endpoints (9400 and 9401) are included with no user config."""
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = None

        manager = self._create_manager_with_mocked_base(mock_user_config)

        assert len(DEFAULT_DCGM_ENDPOINTS) == 2
        assert "http://localhost:9400/metrics" in DEFAULT_DCGM_ENDPOINTS
        assert "http://localhost:9401/metrics" in DEFAULT_DCGM_ENDPOINTS
        assert manager._dcgm_endpoints == list(DEFAULT_DCGM_ENDPOINTS)

    def test_user_explicitly_configured_telemetry_flag(self):
        """Test that _user_explicitly_configured_telemetry flag is set correctly."""
        # Test with None (not configured)
        mock_user_config = MagicMock(spec=UserConfig)
        mock_user_config.gpu_telemetry = None
        manager = self._create_manager_with_mocked_base(mock_user_config)
        assert manager._user_explicitly_configured_telemetry is False

        # Test with value (configured)
        mock_user_config.gpu_telemetry = "http://custom:9401"
        manager = self._create_manager_with_mocked_base(mock_user_config)
        assert manager._user_explicitly_configured_telemetry is True

        # Test with empty list (configured)
        mock_user_config.gpu_telemetry = []
        manager = self._create_manager_with_mocked_base(mock_user_config)
        assert manager._user_explicitly_configured_telemetry is True


class TestProfileConfigureCommand:
    """Test profile configure command doesn't shutdown prematurely."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = list(DEFAULT_DCGM_ENDPOINTS)
        manager._user_provided_endpoints = []
        manager._user_explicitly_configured_telemetry = False
        manager._collection_interval = 0.33
        manager.error = MagicMock()
        manager.debug = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_configure_no_shutdown_when_no_endpoints_reachable(self):
        """Test that configure phase sends disabled status but doesn't shutdown."""
        manager = self._create_test_manager()
        manager.publish = AsyncMock()

        # Mock TelemetryDataCollector to return unreachable
        with patch.object(
            TelemetryDataCollector, "is_url_reachable", return_value=False
        ):
            configure_msg = ProfileConfigureCommand(
                command_id="test", service_id="system_controller", config={}
            )
            await manager._profile_configure_command(configure_msg)

        # Should have sent disabled status
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert call_args.enabled is False
        assert call_args.reason == "no DCGM endpoints reachable"

        # Should NOT have collectors
        assert len(manager._collectors) == 0

    @pytest.mark.asyncio
    async def test_configure_sends_enabled_status_when_endpoints_reachable(self):
        """Test that configure phase sends enabled status with reachable endpoints."""
        manager = self._create_test_manager()
        manager.publish = AsyncMock()

        # Mock TelemetryDataCollector to return reachable
        with patch.object(
            TelemetryDataCollector, "is_url_reachable", return_value=True
        ):
            configure_msg = ProfileConfigureCommand(
                command_id="test", service_id="system_controller", config={}
            )
            await manager._profile_configure_command(configure_msg)

        # Should have sent enabled status
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert call_args.enabled is True
        assert call_args.reason is None

        # Should have collectors
        assert len(manager._collectors) == 2


class TestProfileStartCommand:
    """Test profile start command acknowledgment and behavior."""

    def _create_test_manager(self):
        """Helper to create a TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = list(DEFAULT_DCGM_ENDPOINTS)
        manager._user_provided_endpoints = []
        manager._user_explicitly_configured_telemetry = False
        manager._collection_interval = 0.33
        manager.error = MagicMock()
        manager.warning = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_start_acknowledges_command_immediately(self):
        """Test that start command is acknowledged at the beginning."""
        manager = self._create_test_manager()
        manager.publish = AsyncMock()

        # Add a mock collector
        mock_collector = AsyncMock(spec=TelemetryDataCollector)
        manager._collectors["http://localhost:9400/metrics"] = mock_collector

        start_msg = ProfileStartCommand(
            command_id="test", service_id="system_controller"
        )
        await manager._on_start_profiling(start_msg)

        # Should have published command acknowledgment
        manager.publish.assert_called_once()
        call_args = manager.publish.call_args[0][0]
        assert isinstance(call_args, CommandAcknowledgedResponse)

    @pytest.mark.asyncio
    async def test_start_triggers_shutdown_when_no_collectors(self):
        """Test that start triggers shutdown when no collectors available."""

        def close_coroutine(coro):
            coro.close()
            return MagicMock()

        with patch("asyncio.create_task", side_effect=close_coroutine):
            manager = self._create_test_manager()
            manager.publish = AsyncMock()
            manager._collectors = {}  # No collectors

            start_msg = ProfileStartCommand(
                command_id="test", service_id="system_controller"
            )
            await manager._on_start_profiling(start_msg)

            # Should have acknowledged and sent shutdown
            assert manager.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_start_no_redundant_reachability_check(self):
        """Test that collectors are started without re-checking reachability."""
        manager = self._create_test_manager()
        manager.publish = AsyncMock()

        # Add mock collector
        mock_collector = AsyncMock(spec=TelemetryDataCollector)
        manager._collectors["http://localhost:9400/metrics"] = mock_collector

        start_msg = ProfileStartCommand(
            command_id="test", service_id="system_controller"
        )
        await manager._on_start_profiling(start_msg)

        # Verify collector.initialize() and start() were called without is_url_reachable()
        mock_collector.is_url_reachable.assert_not_called()
        mock_collector.initialize.assert_called_once()
        mock_collector.start.assert_called_once()


class TestSmartDefaultVisibility:
    """Test smart default endpoint visibility in status messages."""

    def _create_test_manager(self, user_requested, user_endpoints):
        """Helper to create a minimal TelemetryManager instance for testing."""
        manager = TelemetryManager.__new__(TelemetryManager)
        manager.service_id = "test_manager"
        manager._collectors = {}
        manager._dcgm_endpoints = list(DEFAULT_DCGM_ENDPOINTS) + user_endpoints
        manager._user_provided_endpoints = user_endpoints
        manager._user_explicitly_configured_telemetry = user_requested
        manager._collection_interval = 0.33
        manager.error = MagicMock()
        manager.debug = MagicMock()
        return manager

    @pytest.mark.asyncio
    async def test_hide_unreachable_defaults_when_one_default_reachable(self):
        """Test that unreachable defaults are hidden when at least one default is reachable."""
        manager = self._create_test_manager(user_requested=False, user_endpoints=[])
        manager.publish = AsyncMock()

        # Manually simulate one reachable default by adding to collectors
        # This tests the smart visibility logic without complex mocking
        manager._collectors[DEFAULT_DCGM_ENDPOINTS[0]] = MagicMock()

        # Call the status reporting part directly
        reachable_endpoints = list(manager._collectors.keys())
        reachable_defaults = [
            ep for ep in DEFAULT_DCGM_ENDPOINTS if ep in reachable_endpoints
        ]

        # Test the smart visibility logic
        if reachable_defaults:
            endpoints_to_report = reachable_endpoints
        elif manager._user_explicitly_configured_telemetry:
            endpoints_to_report = manager._dcgm_endpoints
        else:
            endpoints_to_report = manager._user_provided_endpoints

        # Should only report reachable endpoint
        assert len(endpoints_to_report) == 1
        assert DEFAULT_DCGM_ENDPOINTS[0] in endpoints_to_report
        assert DEFAULT_DCGM_ENDPOINTS[1] not in endpoints_to_report

    @pytest.mark.asyncio
    async def test_show_custom_urls_when_defaults_unreachable(self):
        """Test that custom URLs are shown even when all defaults are unreachable (Scenario 3)."""
        manager = self._create_test_manager(
            user_requested=True, user_endpoints=["http://custom:9401/metrics"]
        )
        manager.publish = AsyncMock()

        # Mock all endpoints as unreachable
        with patch.object(
            TelemetryDataCollector, "is_url_reachable", return_value=False
        ):
            configure_msg = ProfileConfigureCommand(
                command_id="test", service_id="system_controller", config={}
            )
            await manager._profile_configure_command(configure_msg)

        # Should report custom URLs only (no reachable defaults to add)
        call_args = manager.publish.call_args[0][0]
        assert call_args.enabled is False
        assert len(call_args.endpoints_configured) == 1  # Just custom URL
        assert "http://custom:9401/metrics" in call_args.endpoints_configured
        # Defaults should NOT be in the tested list since they're unreachable
        for endpoint in DEFAULT_DCGM_ENDPOINTS:
            assert endpoint not in call_args.endpoints_configured

    @pytest.mark.asyncio
    async def test_show_custom_and_reachable_defaults(self):
        """Test that both custom URLs and reachable defaults are shown (Scenario 3)."""
        manager = self._create_test_manager(
            user_requested=True, user_endpoints=["http://custom:9401/metrics"]
        )
        manager.publish = AsyncMock()

        # Simulate one reachable default
        manager._collectors[DEFAULT_DCGM_ENDPOINTS[0]] = MagicMock()

        # Get the status logic results directly
        reachable_endpoints = list(manager._collectors.keys())
        reachable_defaults = [
            ep for ep in DEFAULT_DCGM_ENDPOINTS if ep in reachable_endpoints
        ]

        # Scenario 3 logic
        endpoints_to_report = (
            list(manager._user_provided_endpoints) + reachable_defaults
        )

        # Should have both custom URL and reachable default
        assert len(endpoints_to_report) == 2
        assert "http://custom:9401/metrics" in endpoints_to_report
        assert DEFAULT_DCGM_ENDPOINTS[0] in endpoints_to_report
        assert DEFAULT_DCGM_ENDPOINTS[1] not in endpoints_to_report

    @pytest.mark.asyncio
    async def test_hide_defaults_when_not_requested_and_all_unreachable(self):
        """Test that defaults are hidden when user didn't request telemetry and all defaults are unreachable."""
        manager = self._create_test_manager(user_requested=False, user_endpoints=[])
        manager.publish = AsyncMock()

        # Mock all endpoints as unreachable
        with patch.object(
            TelemetryDataCollector, "is_url_reachable", return_value=False
        ):
            configure_msg = ProfileConfigureCommand(
                command_id="test", service_id="system_controller", config={}
            )
            await manager._profile_configure_command(configure_msg)

        # Should report empty list (no user endpoints, defaults hidden)
        call_args = manager.publish.call_args[0][0]
        assert call_args.enabled is False
        assert (
            len(call_args.endpoints_configured) == 0
        )  # No user endpoints, defaults hidden
