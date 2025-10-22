# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Annotated

from pydantic import BeforeValidator, Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import DeveloperOnlyCLI
from aiperf.common.config.config_defaults import DevDefaults
from aiperf.common.config.config_validators import parse_service_types
from aiperf.common.constants import AIPERF_DEV_MODE
from aiperf.common.enums.service_enums import ServiceType


def print_developer_mode_warning() -> None:
    """Print a warning message to the console if developer mode is enabled."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    panel = Panel(
        Text(
            "Developer Mode is active. This is a developer-only feature. Use at your own risk.",
            style="yellow",
        ),
        title="AIPerf Developer Mode",
        border_style="bold yellow",
        title_align="left",
    )
    console.print(panel)
    console.file.flush()


if AIPERF_DEV_MODE:
    # Print a warning message to the console if developer mode is enabled, once at load time
    print_developer_mode_warning()


class DeveloperConfig(BaseConfig):
    """
    A configuration class for defining developer related settings.

    NOTE: These settings are only available in developer mode.
    """

    enable_yappi: Annotated[
        bool,
        Field(
            description="*[Developer use only]* Enable yappi profiling (Yet Another Python Profiler) to profile AIPerf's internal python code. "
            "This can be used in the development of AIPerf in order to find performance bottlenecks across the various services. "
            "The output '.prof' files can be viewed with snakeviz. Requires yappi and snakeviz to be installed. "
            "Run 'pip install yappi snakeviz' to install them.",
        ),
        DeveloperOnlyCLI(
            name=("--enable-yappi-profiling"),
        ),
    ] = DevDefaults.ENABLE_YAPPI

    debug_services: Annotated[
        set[ServiceType] | None,
        Field(
            description="*[Developer use only]* List of services to enable debug logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        DeveloperOnlyCLI(
            name=("--debug-service", "--debug-services"),
        ),
        BeforeValidator(parse_service_types),
    ] = DevDefaults.DEBUG_SERVICES

    trace_services: Annotated[
        set[ServiceType] | None,
        Field(
            description="*[Developer use only]* List of services to enable trace logging for. Can be a comma-separated list, a single service type, "
            "or the cli flag can be used multiple times.",
        ),
        DeveloperOnlyCLI(
            name=("--trace-service", "--trace-services"),
        ),
        BeforeValidator(parse_service_types),
    ] = DevDefaults.TRACE_SERVICES

    show_internal_metrics: Annotated[
        bool,
        Field(
            description="*[Developer use only]* Whether to show internal and hidden metrics in the output",
        ),
        DeveloperOnlyCLI(
            name=("--show-internal-metrics"),
        ),
    ] = DevDefaults.SHOW_INTERNAL_METRICS

    disable_uvloop: Annotated[
        bool,
        Field(
            description="*[Developer use only]* Disable the use of uvloop, and use the default asyncio event loop instead.",
        ),
        DeveloperOnlyCLI(
            name=("--disable-uvloop"),
        ),
    ] = DevDefaults.DISABLE_UVLOOP
