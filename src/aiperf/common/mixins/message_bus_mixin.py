# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import ABC
from collections.abc import Callable, Coroutine
from typing import Any

from aiperf.common.config import ServiceConfig
from aiperf.common.constants import (
    DEFAULT_CONNECTION_PROBE_INTERVAL,
    DEFAULT_CONNECTION_PROBE_TIMEOUT,
)
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommAddress
from aiperf.common.enums.message_enums import MessageType
from aiperf.common.hooks import (
    AIPerfHook,
    Hook,
    on_init,
    on_start,
    provides_hooks,
)
from aiperf.common.messages import Message
from aiperf.common.messages.command_messages import ConnectionProbeMessage
from aiperf.common.mixins.communication_mixin import CommunicationMixin
from aiperf.common.protocols import MessageBusClientProtocol
from aiperf.common.types import MessageCallbackMapT, MessageTypeT
from aiperf.common.utils import yield_to_event_loop


@provides_hooks(AIPerfHook.ON_MESSAGE)
@implements_protocol(MessageBusClientProtocol)
class MessageBusClientMixin(CommunicationMixin, ABC):
    """Mixin to provide message bus clients (pub and sub)for AIPerf components, as well as
    a hook to handle messages: @on_message."""

    def __init__(self, service_config: ServiceConfig, **kwargs) -> None:
        super().__init__(service_config=service_config, **kwargs)
        # NOTE: The communication base class will automatically manage the pub/sub clients' lifecycle.
        self.sub_client = self.comms.create_sub_client(
            CommAddress.EVENT_BUS_PROXY_BACKEND
        )
        self.pub_client = self.comms.create_pub_client(
            CommAddress.EVENT_BUS_PROXY_FRONTEND
        )
        self._connection_probe_event = asyncio.Event()

    @on_init
    async def _setup_on_message_hooks(self) -> None:
        """Send subscription requests for all @on_message hook decorators."""
        subscription_map: MessageCallbackMapT = {}

        def _add_to_subscription_map(hook: Hook, message_type: MessageTypeT) -> None:
            """
            This function is called for every message_type parameter of every @on_message hook.
            We use this to build a map of message types to callbacks, which is then used to call
            subscribe_all for efficiency
            """
            self.debug(
                lambda: f"Adding subscription for message type: '{message_type}' for hook: {hook}"
            )
            subscription_map.setdefault(message_type, []).append(hook.func)

        # For each @on_message hook, add each message type to the subscription map.
        self.for_each_hook_param(
            AIPerfHook.ON_MESSAGE,
            self_obj=self,
            param_type=MessageTypeT,
            lambda_func=_add_to_subscription_map,
        )
        self.debug(lambda: f"Subscribing to {len(subscription_map)} topics")
        await self.sub_client.subscribe_all(subscription_map)

        # Subscribe to the connection probe last, to ensure the other subscriptions have been
        # subscribed to before the connection probe is received.
        await self.sub_client.subscribe(
            # NOTE: It is important to use `self.id` here, as not all message bus clients are services
            f"{MessageType.CONNECTION_PROBE}.{self.id}",
            self._process_connection_probe_message,
        )

    @on_start
    async def _wait_for_successful_probe(self) -> None:
        """Send connection probe messages until a successful probe response is received."""
        self.debug(lambda: f"Waiting for connection probe message for {self.id}")

        async def _probe_loop() -> None:
            while not self.stop_requested:
                try:
                    await asyncio.wait_for(
                        self._probe_and_wait_for_response(),
                        timeout=DEFAULT_CONNECTION_PROBE_INTERVAL,
                    )
                    break
                except asyncio.TimeoutError:
                    self.debug(
                        "Timeout waiting for connection probe message, sending another probe"
                    )
                    await yield_to_event_loop()

        await asyncio.wait_for(_probe_loop(), timeout=DEFAULT_CONNECTION_PROBE_TIMEOUT)

    async def _process_connection_probe_message(
        self, message: ConnectionProbeMessage
    ) -> None:
        """Process a connection probe message."""
        self.debug(lambda: f"Received connection probe message: {message}")
        self._connection_probe_event.set()

    async def _probe_and_wait_for_response(self) -> None:
        """Wait for a connection probe message."""
        await self.publish(
            ConnectionProbeMessage(service_id=self.id, target_service_id=self.id)
        )
        await self._connection_probe_event.wait()

    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a specific message type. The callback will be called when
        a message is received for the given message type."""
        await self.sub_client.subscribe(message_type, callback)

    async def subscribe_all(
        self,
        message_callback_map: MessageCallbackMapT,
    ) -> None:
        """Subscribe to all message types in the map. The callback(s) will be called when
        a message is received for the given message type.

        Args:
            message_callback_map: A map of message types to callbacks. The callbacks can be a single callback or a list of callbacks.
        """
        await self.sub_client.subscribe_all(message_callback_map)

    async def publish(self, message: Message) -> None:
        """Publish a message. The message will be routed automatically based on the message type."""
        await self.pub_client.publish(message)
