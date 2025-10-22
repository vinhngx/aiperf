# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC
from collections.abc import Iterable
from typing import Any

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import DEFAULT_COMMAND_RESPONSE_TIMEOUT
from aiperf.common.enums import MessageType
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    on_message,
    provides_hooks,
)
from aiperf.common.messages import (
    CommandAcknowledgedResponse,
    CommandErrorResponse,
    CommandMessage,
    CommandResponse,
    CommandSuccessResponse,
    CommandUnhandledResponse,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import ErrorDetails


@provides_hooks(AIPerfHook.ON_COMMAND)
class CommandHandlerMixin(MessageBusClientMixin, ABC):
    """Mixin to provide command handling functionality to a service.

    This mixin is used by the BaseService class, and is not intended to be used directly.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str,
        **kwargs,
    ) -> None:
        self.service_config = service_config
        self.user_config = user_config
        self.service_id = service_id

        # Keep track of command IDs that have been processed.
        # This is used to avoid processing duplicate command messages.
        self._processed_command_ids: set[str] = set()

        # Keep track of futures for single response commands.
        # This is used to wait for the response from a single service.
        self._single_response_futures: dict[str, asyncio.Future[CommandResponse]] = {}

        # Keep track of futures for multi response commands.
        # This is used to wait for the responses from multiple services.
        self._multi_response_futures: dict[
            str, dict[str, asyncio.Future[CommandResponse]]
        ] = {}

        super().__init__(
            service_config=self.service_config,
            user_config=self.user_config,
            **kwargs,
        )

    @on_message(
        lambda self: {
            # Subscribe to all broadcast command messages.
            MessageType.COMMAND,
            # Subscribe to all command messages for this specific service type.
            f"{MessageType.COMMAND}.{self.service_type}",
            # Subscribe to all command messages for this specific service ID.
            f"{MessageType.COMMAND}.{self.service_id}",
        }
    )
    async def _process_command_message(self, message: CommandMessage) -> None:
        """
        Process a command message received from the controller or another service, and forward it to the appropriate handler.
        Wait for the handler to complete and publish the response, or handle the error and publish the failure response.
        """
        self.debug(lambda: f"Received command message: {message}")
        if message.command_id in self._processed_command_ids:
            self.debug(
                lambda: f"Received duplicate command message: {message}. Ignoring."
            )
            # If we receive a duplicate command message, we just send an acknowledged response.
            await self._publish_command_acknowledged_response(message)
            return

        self._processed_command_ids.add(message.command_id)

        if message.service_id == self.service_id:
            # In the case of a broadcast command, you will receive a command message from yourself.
            # We ignore these messages.
            self.debug(
                lambda: f"Received broadcast command message from self: {message}. Ignoring."
            )
            return

        # Go through the hooks and find the first one that matches the command type.
        # Currently, we only support a single handler per command type, so we break out of the loop after the first one.
        # The reason for this is because we are sending the result of the handler function back to the original service that sent the command.
        # If there were multiple handlers, we would need to handle multiple responses, partial errors, etc.
        # TODO: Do we want/need to add support for multiple handlers per command type?
        for hook in self.get_hooks(AIPerfHook.ON_COMMAND):
            if isinstance(hook.params, Iterable) and message.command in hook.params:
                await self._execute_command_hook(message, hook)
                # Only one handler per command type, so return after the first handler.
                return

        # If we reach here, no handler was found for the command, so we publish an unhandled response.
        await self._publish_command_unhandled_response(message)

    async def _execute_command_hook(self, message: CommandMessage, hook: Hook) -> None:
        """Execute a command hook.
        This is the internal function that wraps calling a hook function and publishing the response
        based on the return value of the hook function.
        """
        try:
            result = await hook.func(message)
            if result is None:
                # If there is no data to send back, just send an acknowledged response.
                await self._publish_command_acknowledged_response(message)
                return
            await self._publish_command_success_response(message, result)
        except Exception as e:
            self.exception(
                f"Failed to handle command {message.command} with hook {hook}: {e}"
            )
            await self._publish_command_error_response(
                message, ErrorDetails.from_exception(e)
            )

    async def _publish_command_acknowledged_response(
        self, message: CommandMessage
    ) -> None:
        """Publish a command acknowledged response to a command message."""
        await self.publish(
            CommandAcknowledgedResponse.from_command_message(message, self.service_id)
        )

    async def _publish_command_success_response(
        self, message: CommandMessage, result: Any
    ) -> None:
        """Publish a command success response to a command message."""
        await self.publish(
            CommandSuccessResponse.from_command_message(
                message, self.service_id, result
            )
        )

    async def _publish_command_error_response(
        self, message: CommandMessage, error: ErrorDetails
    ) -> None:
        """Publish a command error response to a command message."""
        await self.publish(
            CommandErrorResponse.from_command_message(message, self.service_id, error)
        )

    async def _publish_command_unhandled_response(
        self, message: CommandMessage
    ) -> None:
        """Publish a command unhandled response to a command message."""
        await self.publish(
            CommandUnhandledResponse.from_command_message(message, self.service_id)
        )

    async def send_command_and_wait_for_response(
        self, message: CommandMessage, timeout: float = DEFAULT_COMMAND_RESPONSE_TIMEOUT
    ) -> CommandResponse | ErrorDetails:
        """Send a single command message to a single service and wait for the response.
        This is useful communicating directly with a single service.
        """
        # Create a future that we can asynchronously wait for the response.
        future = asyncio.Future[CommandResponse]()
        self._single_response_futures[message.command_id] = future
        await self.publish(message)
        try:
            # Wait for the response future to be set by the command response message handler.
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            return ErrorDetails.from_exception(e)
        finally:
            future.cancel()
            del self._single_response_futures[message.command_id]

    async def send_command_and_wait_for_all_responses(
        self,
        command: CommandMessage,
        service_ids: list[str],
        timeout: float = DEFAULT_COMMAND_RESPONSE_TIMEOUT,
    ) -> list[CommandResponse | ErrorDetails]:
        """Broadcast a command message to multiple services and wait for the responses from all of the specified services.
        This is useful for the system controller to send one command but wait for all services to respond.
        """
        # Create a future to track the response for each service ID.
        self._multi_response_futures[command.command_id] = {
            service_id: asyncio.Future[CommandResponse]() for service_id in service_ids
        }
        # Send the command to all services based on the target service ID and target service type.
        await self.publish(command)
        try:
            # Wait for all the responses to come in.
            return await asyncio.wait_for(
                asyncio.gather(
                    *[
                        future
                        for future in self._multi_response_futures[
                            command.command_id
                        ].values()
                    ]
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            return [ErrorDetails.from_exception(e) for _ in range(len(service_ids))]
        finally:
            # Clean up the response futures.
            for future in self._multi_response_futures[command.command_id].values():
                future.cancel()
            del self._multi_response_futures[command.command_id]

    @on_message(
        lambda self: {
            # NOTE: Command responses are only ever sent to the original service that sent the command,
            #       so we only need to subscribe to the service ID specific topic.
            f"{MessageType.COMMAND_RESPONSE}.{self.service_id}",
        }
    )
    async def _process_command_response_message(self, message: CommandResponse) -> None:
        """
        Process a command response message received from a service. This function is called whenever
        a command response is received, and we use it to set the result of the future for the command ID.
        This will alert the the task that is waiting for the response to continue.
        """
        self.debug(lambda: f"Received command response message: {message}")

        # If the command ID is in the single response futures, we set the result of the future.
        # This will alert the the task that is waiting for the response to continue.
        if message.command_id in self._single_response_futures:
            self._single_response_futures[message.command_id].set_result(message)
            return

        # If the command ID is in the multi response futures, we set the result of the future for the service ID of the sender.
        # This will alert the the task that is waiting for the response to continue.
        if message.command_id in self._multi_response_futures:
            if message.service_id in self._multi_response_futures[message.command_id]:
                self._multi_response_futures[message.command_id][
                    message.service_id
                ].set_result(message)
            else:
                self.warning(
                    f"Received command response for service we were not expecting: {message.service_id}. Ignoring."
                )
            return

        # If we reach here, we received a command response that we were not tracking. It is
        # safe to ignore.
        self.debug(
            lambda: f"Received command response for untracked command: {message}. Ignoring."
        )
