# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for dataset manager testing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import aiperf.endpoints  # noqa: F401  # Import to register endpoints
import aiperf.transports  # noqa: F401  # Import to register transports
from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import Conversation
from aiperf.dataset.dataset_manager import DatasetManager


@pytest.fixture
def user_config(tmp_path: Path) -> UserConfig:
    """Create a UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            streaming=False,
            url="http://localhost:8000",
        ),
        output=OutputConfig(artifact_directory=tmp_path),
    )


@pytest.fixture
def empty_dataset_manager(
    user_config: UserConfig,
) -> DatasetManager:
    """Create a DatasetManager instance with empty dataset."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = {}
    return manager


@pytest.fixture
def populated_dataset_manager(
    user_config: UserConfig,
    sample_conversations: dict[str, Conversation],
) -> DatasetManager:
    """Create a DatasetManager instance with sample data."""
    manager = DatasetManager(
        service_config=ServiceConfig(),
        user_config=user_config,
        service_id="test_dataset_manager",
    )
    manager.dataset = sample_conversations
    return manager


@pytest.fixture
def capture_file_writes():
    """Provide a fixture to capture file write operations for testing purposes."""

    class FileWriteCapture:
        def __init__(self):
            self.written_content = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def write(self, content: str):
            self.written_content = content

    class AsyncContextManager:
        def __init__(self, capture):
            self.capture = capture

        async def __aenter__(self):
            return self.capture

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    capture = FileWriteCapture()

    def mock_open(file_path, mode="r"):
        return AsyncContextManager(capture)

    with patch("aiperf.dataset.dataset_manager.aiofiles.open", mock_open):
        yield capture
