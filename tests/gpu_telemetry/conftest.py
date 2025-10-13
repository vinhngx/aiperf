# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures for testing GPU telemetry components.
"""

import pytest

from aiperf.common.models.telemetry_models import TelemetryMetrics, TelemetryRecord


@pytest.fixture
def sample_dcgm_data():
    """Sample DCGM metrics data in Prometheus format (single GPU)."""

    return """# HELP DCGM_FI_DEV_SM_CLOCK SM clock frequency (in MHz)
# TYPE DCGM_FI_DEV_SM_CLOCK gauge
DCGM_FI_DEV_SM_CLOCK{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 210
# HELP DCGM_FI_DEV_MEM_CLOCK Memory clock frequency (in MHz)
# TYPE DCGM_FI_DEV_MEM_CLOCK gauge
DCGM_FI_DEV_MEM_CLOCK{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 405
# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W)
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 22.582000
# HELP DCGM_FI_DEV_POWER_MGMT_LIMIT Power management limit (in W)
# TYPE DCGM_FI_DEV_POWER_MGMT_LIMIT gauge
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 300.0
# HELP DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION Total energy consumption since boot (in mJ)
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 955287014
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %)
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 1
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB)
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 46614
# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (in MiB)
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 2048
# HELP DCGM_FI_DEV_FB_TOTAL Total framebuffer memory (in MiB)
# TYPE DCGM_FI_DEV_FB_TOTAL gauge
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 48662
# HELP DCGM_FI_DEV_MEM_COPY_UTIL Memory copy utilization (in %)
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 15
# HELP DCGM_FI_DEV_XID_ERRORS Value of the last XID error encountered
# TYPE DCGM_FI_DEV_XID_ERRORS gauge
DCGM_FI_DEV_XID_ERRORS{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
# HELP DCGM_FI_DEV_POWER_VIOLATION Throttling duration due to power constraints (in us)
# TYPE DCGM_FI_DEV_POWER_VIOLATION counter
DCGM_FI_DEV_POWER_VIOLATION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 12000
# HELP DCGM_FI_DEV_THERMAL_VIOLATION Throttling duration due to thermal constraints (in us)
# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter
DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 5000
"""


@pytest.fixture
def multi_gpu_dcgm_data():
    """Sample DCGM metrics data with multiple GPUs."""

    return """# HELP DCGM_FI_DEV_POWER_USAGE Power draw (in W)
# TYPE DCGM_FI_DEV_POWER_USAGE gauge
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 79.60
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 42.09
DCGM_FI_DEV_POWER_USAGE{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 43.99
# HELP DCGM_FI_DEV_POWER_MGMT_LIMIT Power management limit (in W)
# TYPE DCGM_FI_DEV_POWER_MGMT_LIMIT gauge
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 300.0
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 300.0
DCGM_FI_DEV_POWER_MGMT_LIMIT{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 700.0
# HELP DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION Total energy consumption since boot (in mJ)
# TYPE DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION counter
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 280000000
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 230000000
DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 250000000
# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %)
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 34
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_GPU_UTIL{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 0
# HELP DCGM_FI_DEV_FB_USED Framebuffer memory used (in MiB)
# TYPE DCGM_FI_DEV_FB_USED gauge
DCGM_FI_DEV_FB_USED{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 15640
DCGM_FI_DEV_FB_USED{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_FB_USED{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 0
# HELP DCGM_FI_DEV_FB_FREE Framebuffer memory free (in MiB)
# TYPE DCGM_FI_DEV_FB_FREE gauge
DCGM_FI_DEV_FB_FREE{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 33022
DCGM_FI_DEV_FB_FREE{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 48662
DCGM_FI_DEV_FB_FREE{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 81920
# HELP DCGM_FI_DEV_FB_TOTAL Total framebuffer memory (in MiB)
# TYPE DCGM_FI_DEV_FB_TOTAL gauge
DCGM_FI_DEV_FB_TOTAL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 48662
DCGM_FI_DEV_FB_TOTAL{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 48662
DCGM_FI_DEV_FB_TOTAL{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 81920
# HELP DCGM_FI_DEV_MEM_COPY_UTIL Memory copy utilization (in %)
# TYPE DCGM_FI_DEV_MEM_COPY_UTIL gauge
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 20
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_MEM_COPY_UTIL{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 10
# HELP DCGM_FI_DEV_XID_ERRORS Value of the last XID error encountered
# TYPE DCGM_FI_DEV_XID_ERRORS gauge
DCGM_FI_DEV_XID_ERRORS{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_XID_ERRORS{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_XID_ERRORS{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 0
# HELP DCGM_FI_DEV_POWER_VIOLATION Throttling duration due to power constraints (in us)
# TYPE DCGM_FI_DEV_POWER_VIOLATION counter
DCGM_FI_DEV_POWER_VIOLATION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 15000
DCGM_FI_DEV_POWER_VIOLATION{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_POWER_VIOLATION{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 8000
# HELP DCGM_FI_DEV_THERMAL_VIOLATION Throttling duration due to thermal constraints (in us)
# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter
DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0",UUID="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",pci_bus_id="00000000:02:00.0",device="nvidia0",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 3000
DCGM_FI_DEV_THERMAL_VIOLATION{gpu="1",UUID="GPU-12345678-1234-1234-1234-123456789abc",pci_bus_id="00000000:03:00.0",device="nvidia1",modelName="NVIDIA RTX 6000 Ada Generation",Hostname="ed7e7a5e585f"} 0
DCGM_FI_DEV_THERMAL_VIOLATION{gpu="2",UUID="GPU-87654321-4321-4321-4321-cba987654321",pci_bus_id="00000000:04:00.0",device="nvidia2",modelName="NVIDIA H100 PCIe",Hostname="ed7e7a5e585f"} 5000
"""


@pytest.fixture
def sample_telemetry_records():
    """Sample TelemetryRecord objects for testing."""

    return [
        TelemetryRecord(
            timestamp_ns=1000000000,
            dcgm_url="http://localhost:9401/metrics",
            gpu_index=0,
            gpu_model_name="NVIDIA RTX 6000 Ada Generation",
            gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
            pci_bus_id="00000000:02:00.0",
            device="nvidia0",
            hostname="ed7e7a5e585f",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=22.582,
                energy_consumption=955.287014,
                gpu_utilization=1.0,
                gpu_memory_used=45.521,  # 46614 MiB / 1024 ≈ 45.521 GB
            ),
        ),
    ]


@pytest.fixture
def multi_gpu_telemetry_records():
    """Multiple GPU records for batch processing tests."""

    records = []

    # Generate 50 samples for each GPU with realistic patterns
    for i in range(50):
        timestamp = 1000000000 + (i * 33000000)  # 33ms intervals

        # GPU 0 - Active workload (RTX 6000)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=0,
                gpu_model_name="NVIDIA RTX 6000 Ada Generation",
                gpu_uuid="GPU-ef6ef310-f8e2-cef9-036e-8f12d59b5ffc",
                pci_bus_id="00000000:02:00.0",
                device="nvidia0",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=70.0 + (i % 30),  # Varying power 70-99W
                    energy_consumption=(280000000 + (i * 2000000))
                    / 1e6,  # Increasing energy
                    gpu_utilization=float(80 + (i % 20)),  # 80-99%
                    gpu_memory_used=15.0 + (i % 5),  # 15-19 GB
                ),
            )
        )

        # GPU 1 - Idle (RTX 6000)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp + 1000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=1,
                gpu_model_name="NVIDIA RTX 6000 Ada Generation",
                gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                pci_bus_id="00000000:03:00.0",
                device="nvidia1",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=42.0 + (i % 3),  # Idle power 42-44W
                    energy_consumption=(230000000 + (i * 500000))
                    / 1e6,  # Slower energy growth
                    gpu_utilization=0.0,
                    gpu_memory_used=0.0,
                ),
            )
        )

        # GPU 2 - Moderate workload (H100)
        records.append(
            TelemetryRecord(
                timestamp_ns=timestamp + 2000,
                dcgm_url="http://localhost:9401/metrics",
                gpu_index=2,
                gpu_model_name="NVIDIA H100 PCIe",
                gpu_uuid="GPU-87654321-4321-4321-4321-cba987654321",
                pci_bus_id="00000000:04:00.0",
                device="nvidia2",
                hostname="ed7e7a5e585f",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=200.0 + (i % 50),  # Higher power 200-249W
                    energy_consumption=(250000000 + (i * 3000000)) / 1e6,
                    gpu_utilization=float(50 + (i % 30)),  # 50-79%
                    gpu_memory_used=40.0 + (i % 10),  # 40-49 GB
                ),
            )
        )

    return records
