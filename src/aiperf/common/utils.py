# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
import os
import traceback
from collections.abc import Callable
from typing import Any

import orjson

from aiperf.common import aiperf_logger
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.exceptions import AIPerfMultiError

_logger = AIPerfLogger(__name__)


async def call_all_functions_self(
    self_: object, funcs: list[Callable], *args, **kwargs
) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(self_, *args, **kwargs)
            else:
                func(self_, *args, **kwargs)
        except Exception as e:
            # TODO: error handling, logging
            traceback.print_exc()
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


async def call_all_functions(funcs: list[Callable], *args, **kwargs) -> None:
    """Call all functions in the list with the given name.

    Args:
        obj: The object to call the functions on.
        func_names: The names of the functions to call.
        *args: The arguments to pass to the functions.
        **kwargs: The keyword arguments to pass to the functions.

    Raises:
        AIPerfMultiError: If any of the functions raise an exception.
    """

    exceptions = []
    for func in funcs:
        try:
            if inspect.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        except Exception as e:
            # TODO: error handling, logging
            traceback.print_exc()
            exceptions.append(e)

    if len(exceptions) > 0:
        raise AIPerfMultiError("Errors calling functions", exceptions)


def load_json_str(json_str: str, func: Callable = lambda x: x) -> dict[str, Any]:
    """
    Deserializes JSON encoded string into Python object.

    Args:
      - json_str: string
          JSON encoded string
      - func: callable
          A function that takes deserialized JSON object. This can be used to
          run validation checks on the object. Defaults to identity function.
    """
    try:
        # Note: orjson may not parse JSON the same way as Python's standard json library,
        # notably being stricter on UTF-8 conformance.
        # Refer to https://github.com/ijl/orjson?tab=readme-ov-file#str for details.
        return func(orjson.loads(json_str))
    except orjson.JSONDecodeError as e:
        snippet = json_str[:200] + ("..." if len(json_str) > 200 else "")
        _logger.exception(f"Failed to parse JSON string: '{snippet}' - {e!r}")
        raise


async def yield_to_event_loop() -> None:
    """Yield to the event loop. This forces the current coroutine to yield and allow
    other coroutines to run, preventing starvation. Use this when you do not want to
    delay your coroutine via sleep, but still want to allow other coroutines to run if
    there is a potential for an infinite loop.
    """
    await asyncio.sleep(0)


def compute_time_ns(
    start_time_ns: int, start_perf_ns: int, perf_ns: int | None
) -> int | None:
    """Convert a perf_ns timestamp to a wall clock time_ns timestamp by
    computing the absolute duration in perf_ns (perf_ns - start_perf_ns) and adding it to the start_time_ns.

    Args:
        start_time_ns: The wall clock start time in nanoseconds (time.time_ns).
        start_perf_ns: The start perf time in nanoseconds (perf_counter_ns).
        perf_ns: The perf time in nanoseconds to convert to time_ns (perf_counter_ns).

    Returns:
        The perf_ns converted to time_ns, or None if the perf_ns is None.
    """
    if perf_ns is None:
        return None
    if perf_ns < start_perf_ns:
        raise ValueError(f"perf_ns {perf_ns} is before start_perf_ns {start_perf_ns}")
    return start_time_ns + (perf_ns - start_perf_ns)


# This is used to identify the source file of the call_all_functions function
# in the AIPerfLogger class to skip it when determining the caller.
# NOTE: Using similar logic to logging._srcfile
_srcfile = os.path.normcase(call_all_functions.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)
