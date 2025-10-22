# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
import time
import typing
from typing import Any

import aiohttp

from aiperf.common.enums import SSEFieldType
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    ErrorDetails,
    RequestRecord,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.transports.http_defaults import AioHttpDefaults, SocketDefaults

################################################################################
# AioHTTP Client
################################################################################


class AioHttpClientMixin(AIPerfLoggerMixin):
    """A high-performance HTTP client for communicating with HTTP based REST APIs using aiohttp.

    This class is optimized for maximum performance and accurate timing measurements,
    making it ideal for benchmarking scenarios.
    """

    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs) -> None:
        self.model_endpoint = model_endpoint
        super().__init__(model_endpoint=model_endpoint, **kwargs)
        self.tcp_connector = create_tcp_connector()

        # For now, just set all timeouts to the same value.
        # TODO: Add support for different timeouts for different parts of the request.
        self.timeout = aiohttp.ClientTimeout(
            total=self.model_endpoint.endpoint.timeout,
            connect=self.model_endpoint.endpoint.timeout,
            sock_connect=self.model_endpoint.endpoint.timeout,
            sock_read=self.model_endpoint.endpoint.timeout,
            ceil_threshold=self.model_endpoint.endpoint.timeout,
        )

    async def close(self) -> None:
        """Close the client."""
        if self.tcp_connector:
            await self.tcp_connector.close()
            self.tcp_connector = None

    async def _request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: str | None = None,
        **kwargs: Any,
    ) -> RequestRecord:
        """Generic request method that handles common logic for all HTTP methods.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL to send the request to
            headers: Request headers
            data: Request payload (for POST, PUT, etc.)
            **kwargs: Additional arguments to pass to the request

        Returns:
            RequestRecord with the response data
        """
        self.debug(lambda: f"Sending {method} request to {url}")

        record: RequestRecord = RequestRecord(
            start_perf_ns=time.perf_counter_ns(),
        )

        try:
            # Make raw HTTP request with precise timing using aiohttp
            async with aiohttp.ClientSession(
                connector=self.tcp_connector,
                timeout=self.timeout,
                headers=headers,
                skip_auto_headers=[
                    *list(headers.keys()),
                    "User-Agent",
                    "Accept-Encoding",
                ],
                connector_owner=False,
            ) as session:
                record.start_perf_ns = time.perf_counter_ns()
                async with session.request(
                    method, url, data=data, headers=headers, **kwargs
                ) as response:
                    record.status = response.status
                    # Check for HTTP errors
                    if response.status != 200:
                        error_text = await response.text()
                        record.error = ErrorDetails(
                            code=response.status,
                            type=response.reason,
                            message=error_text,
                        )
                        return record

                    record.recv_start_perf_ns = time.perf_counter_ns()

                    if (
                        method == "POST"
                        and response.content_type == "text/event-stream"
                    ):
                        # Parse SSE stream with optimal performance
                        messages = await AioHttpSSEStreamReader(
                            response
                        ).read_complete_stream()
                        record.responses.extend(messages)
                    else:
                        raw_response = await response.text()
                        record.end_perf_ns = time.perf_counter_ns()
                        record.responses.append(
                            TextResponse(
                                perf_ns=record.end_perf_ns,
                                content_type=response.content_type,
                                text=raw_response,
                            )
                        )
                    record.end_perf_ns = time.perf_counter_ns()

        except Exception as e:
            record.end_perf_ns = time.perf_counter_ns()
            self.error(f"Error in aiohttp request: {e!r}")
            record.error = ErrorDetails(type=e.__class__.__name__, message=str(e))

        return record

    async def post_request(
        self,
        url: str,
        payload: str,
        headers: dict[str, str],
        **kwargs: Any,
    ) -> RequestRecord:
        """Send a streaming or non-streaming POST request to the specified URL with the given payload and headers.

        If the response is an SSE stream, the response will be parsed into a list of SSE messages.
        Otherwise, the response will be parsed into a TextResponse object.
        """
        return await self._request("POST", url, headers, data=payload, **kwargs)

    async def get_request(
        self, url: str, headers: dict[str, str], **kwargs: Any
    ) -> RequestRecord:
        """Send a GET request to the specified URL with the given headers.

        The response will be parsed into a TextResponse object.
        """
        return await self._request("GET", url, headers, **kwargs)


class AioHttpSSEStreamReader:
    """A helper class for reading an SSE stream from an aiohttp.ClientResponse object.

    This class is optimized for maximum performance and accurate timing measurements,
    making it ideal for benchmarking scenarios.
    """

    def __init__(self, response: aiohttp.ClientResponse):
        self.response = response

    async def read_complete_stream(self) -> list[SSEMessage]:
        """Read the complete SSE stream in a performant manner and return a list of
        SSE messages that contain the most accurate timestamp data possible.

        Returns:
            A list of SSE messages.
        """
        messages: list[SSEMessage] = []

        async for raw_message, first_byte_ns in self.__aiter__():
            # Parse the raw SSE message into a SSEMessage object
            message = parse_sse_message(raw_message, first_byte_ns)
            messages.append(message)

        return messages

    async def __aiter__(self) -> typing.AsyncIterator[tuple[str, int]]:
        """Iterate over the SSE stream in a performant manner and return a tuple of the
        raw SSE message, the perf_counter_ns of the first byte, and the perf_counter_ns of the last byte.
        This provides the most accurate timing information possible without any delays due to the nature of
        the aiohttp library. The first byte is read immediately to capture the timestamp of the first byte,
        and the last byte is read after the rest of the chunk is read to capture the timestamp of the last byte.

        Returns:
            An async iterator of tuples of the raw SSE message, and the perf_counter_ns of the first byte
        """

        while not self.response.content.at_eof():
            # Read the first byte of the SSE stream
            first_byte = await self.response.content.read(1)
            chunk_ns_first_byte = time.perf_counter_ns()
            if not first_byte:
                break

            # Read until we hit \n\n which delimits SSE messages
            chunk = await self.response.content.readuntil(b"\n\n")

            if not chunk:
                break
            chunk = first_byte + chunk

            try:
                decoded = chunk.decode("utf-8")
                for sub_chunk in decoded.split("\n\n"):
                    if sub_chunk:
                        yield (sub_chunk, chunk_ns_first_byte)
                # Use the fastest available decoder
            except UnicodeDecodeError:
                # Handle potential encoding issues gracefully
                yield (
                    chunk.decode("utf-8", errors="replace").strip(),
                    chunk_ns_first_byte,
                )


def parse_sse_message(raw_message: str, perf_ns: int) -> SSEMessage:
    """Parse a raw SSE message into an SSEMessage object.

    Parsing logic based on official HTML SSE Living Standard:
    https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
    """

    message = SSEMessage(perf_ns=perf_ns)
    for line in raw_message.split("\n"):
        if not (line := line.strip()):
            continue

        parts = line.split(":", 1)
        if len(parts) < 2:
            # Fields without a colon have no value, so the whole line is the field name
            message.packets.append(SSEField(name=parts[0].strip(), value=None))
            continue

        field_name, value = parts

        if field_name == "":
            # Field name is empty, so this is a comment
            field_name = SSEFieldType.COMMENT

        message.packets.append(SSEField(name=field_name.strip(), value=value.strip()))

    return message


def create_tcp_connector(**kwargs) -> aiohttp.TCPConnector:
    """Create a new connector with the given configuration."""

    def socket_factory(addr_info):
        """Custom socket factory optimized for SSE streaming performance."""
        family, sock_type, proto, _, _ = addr_info
        sock = socket.socket(family=family, type=sock_type, proto=proto)
        SocketDefaults.apply_to_socket(sock)
        return sock

    default_kwargs: dict[str, Any] = {
        "limit": AioHttpDefaults.LIMIT,
        "limit_per_host": AioHttpDefaults.LIMIT_PER_HOST,
        "ttl_dns_cache": AioHttpDefaults.TTL_DNS_CACHE,
        "use_dns_cache": AioHttpDefaults.USE_DNS_CACHE,
        "enable_cleanup_closed": AioHttpDefaults.ENABLE_CLEANUP_CLOSED,
        "force_close": AioHttpDefaults.FORCE_CLOSE,
        "keepalive_timeout": AioHttpDefaults.KEEPALIVE_TIMEOUT,
        "happy_eyeballs_delay": AioHttpDefaults.HAPPY_EYEBALLS_DELAY,
        "family": AioHttpDefaults.SOCKET_FAMILY,
        "socket_factory": socket_factory,
    }

    default_kwargs.update(kwargs)

    return aiohttp.TCPConnector(
        **default_kwargs,
    )
