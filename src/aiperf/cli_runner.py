# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

from aiperf.cli_utils import raise_startup_error_and_exit
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums.ui_enums import AIPerfUIType


def run_system_controller(
    user_config: UserConfig,
    service_config: ServiceConfig,
) -> None:
    """Run the system controller with the given configuration."""

    # NOTE: On macOS, when using the Textual UI with multiprocessing, terminal corruption
    # (ASCII garbage, freezing) can occur when mouse events interfere with child processes.
    # We apply multiple layers of protection:
    # 1. Set spawn method early (before any multiprocessing operations)
    # 2. Create log_queue before any UI initialization
    # 3. Set FD_CLOEXEC on terminal file descriptors
    # 4. Close terminal FDs in child processes (done in bootstrap.py)

    import multiprocessing
    import platform

    is_macos = platform.system() == "Darwin"
    using_dashboard = service_config.ui_type == AIPerfUIType.DASHBOARD

    # Force spawn method on macOS to prevent fork-related issues.
    # This should already be the default, but we'll set it explicitly just in case.
    if is_macos and using_dashboard:
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method("spawn", force=True)

    from aiperf.common.aiperf_logger import AIPerfLogger
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.controller import SystemController
    from aiperf.module_loader import ensure_modules_loaded

    logger = AIPerfLogger(__name__)

    # Create log_queue before UI initialization to minimize FD inheritance issues.
    log_queue = None
    if using_dashboard:
        from aiperf.common.logging import get_global_log_queue

        log_queue = get_global_log_queue()

        # Set FD_CLOEXEC on terminal file descriptors on macOS.
        # This ensures terminal FDs are closed when child processes spawn.
        if is_macos:
            import fcntl
            import sys

            try:
                for fd in [
                    sys.stdin.fileno(),
                    sys.stdout.fileno(),
                    sys.stderr.fileno(),
                ]:
                    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
                    fcntl.fcntl(fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)
                logger.debug("Set FD_CLOEXEC on terminal file descriptors for macOS")
            except (OSError, ValueError, AttributeError) as e:
                # Non-fatal if this fails, other layers will protect
                logger.debug(f"Could not set FD_CLOEXEC on terminal descriptors: {e}")
    else:
        from aiperf.common.logging import setup_rich_logging

        setup_rich_logging(user_config, service_config)

    # Create and start the system controller
    logger.info("Starting AIPerf System")

    try:
        ensure_modules_loaded()
    except Exception as e:
        raise_startup_error_and_exit(
            f"Error loading modules: {e}",
            title="Error Loading Modules",
        )

    try:
        bootstrap_and_run_service(
            SystemController,
            service_id="system_controller",
            service_config=service_config,
            user_config=user_config,
            log_queue=log_queue,
        )
    except Exception:
        logger.exception("Error running AIPerf System")
        raise
    finally:
        logger.debug("AIPerf System exited")
        if log_queue is not None:
            from aiperf.common.logging import cleanup_global_log_queue

            cleanup_global_log_queue()
