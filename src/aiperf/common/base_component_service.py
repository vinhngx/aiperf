# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_HEARTBEAT_INTERVAL,
    DEFAULT_MAX_REGISTRATION_ATTEMPTS,
    DEFAULT_REGISTRATION_INTERVAL,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommandType, LifecycleState, ServiceType
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_start,
    on_state_change,
)
from aiperf.common.messages import (
    CommandMessage,
    HeartbeatMessage,
    StatusMessage,
)
from aiperf.common.messages.command_messages import (
    CommandResponse,
    RegisterServiceCommand,
)
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.protocols import ServiceProtocol


@implements_protocol(ServiceProtocol)
class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by adding heartbeat and registration functionality, as well as
    publishing the current state of the service to the system controller.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

    @background_task(interval=DEFAULT_HEARTBEAT_INTERVAL, immediate=False)
    async def _heartbeat_task(self) -> None:
        """Send a heartbeat notification to the system controller."""
        await self.publish(
            HeartbeatMessage(
                service_id=self.service_id,
                service_type=self.service_type,
                state=self.state,
            )
        )

    @on_start
    async def _register_service_on_start(self) -> None:
        """Register the service with the system controller on startup."""
        self.debug(
            lambda: f"Attempting to register service {self} ({self.service_id}) with system controller"
        )
        result = None
        command_message = RegisterServiceCommand(
            command_id=str(uuid.uuid4()),
            service_id=self.service_id,
            service_type=self.service_type,
            # Target the system controller directly to avoid broadcasting to all services.
            target_service_type=ServiceType.SYSTEM_CONTROLLER,
            state=self.state,
        )
        for _ in range(DEFAULT_MAX_REGISTRATION_ATTEMPTS):
            result = await self.send_command_and_wait_for_response(
                # NOTE: We keep the command id the same each time to ensure that the system controller
                #       can ignore duplicate registration requests.
                command_message,
                timeout=DEFAULT_REGISTRATION_INTERVAL,
            )
            if isinstance(result, CommandResponse):
                self.debug(
                    lambda: f"Service {self.service_id} registered with system controller"
                )
                break
        if isinstance(result, ErrorDetails):
            self.error(
                f"Failed to register service {self} ({self.service_id}): {result}"
            )
            raise self._service_error(
                f"Failed to register service {self} ({self.service_id}): {result}"
            )

    @on_state_change
    async def _on_state_change(
        self, old_state: LifecycleState, new_state: LifecycleState
    ) -> None:
        """Action to take when the service state is set.

        This method will also publish the status message to the status message_type if the
        communications are initialized.
        """
        if self.stop_requested:
            return
        if not self.comms.was_initialized:
            return
        await self.publish(
            StatusMessage(
                service_id=self.service_id,
                service_type=self.service_type,
                state=new_state,
            )
        )

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        self.debug(f"Received shutdown command: {message}, {self.service_id}")
        try:
            await self.stop()
        except Exception as e:
            self.warning(
                f"Failed to stop service {self} ({self.service_id}) after receiving shutdown command: {e}. Killing."
            )
            await self._kill()
        raise asyncio.CancelledError()
