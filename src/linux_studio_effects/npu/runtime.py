"""NPU hardware, kernel driver, and IRON tensor runtime detection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NpuStatus:
    """Result of probing the NPU stack from /dev/accel up to IRON."""

    available: bool
    hardware_detail: str
    driver_detail: str
    device_name: str
    backend: str
    tensor_type: str
    reason: str = ""


def probe_npu() -> NpuStatus:
    """Probe hardware, amdxdna kernel driver, and active IRON NPU tensor allocation."""
    import aie.iron as iron
    from aie.utils import probe
    from aie.utils.hostruntime.tensor_class import NpuTensor

    hw = probe.check_hardware()
    drv = probe.check_driver()
    backend = os.environ.get("NPU_RUNTIME", "xrt")

    if not hw.ok:
        return NpuStatus(
            available=False,
            hardware_detail=hw.detail,
            driver_detail=drv.detail,
            device_name="none",
            backend=backend,
            tensor_type="none",
            reason=hw.remedy or hw.detail,
        )

    if not drv.ok:
        return NpuStatus(
            available=False,
            hardware_detail=hw.detail,
            driver_detail=drv.detail,
            device_name="none",
            backend=backend,
            tensor_type="none",
            reason=drv.remedy or drv.detail,
        )

    device = iron.ensure_current_device()
    if device is None:
        return NpuStatus(
            available=False,
            hardware_detail=hw.detail,
            driver_detail=drv.detail,
            device_name="none",
            backend=backend,
            tensor_type="none",
            reason="iron.ensure_current_device() returned None",
        )

    t = iron.zeros(16, dtype=np.int32, device="npu")
    tensor_type = type(t).__name__
    if not isinstance(t, NpuTensor):
        return NpuStatus(
            available=False,
            hardware_detail=hw.detail,
            driver_detail=drv.detail,
            device_name=repr(device),
            backend=backend,
            tensor_type=tensor_type,
            reason=f"allocated {tensor_type} instead of NpuTensor (CPU fallback)",
        )

    return NpuStatus(
        available=True,
        hardware_detail=hw.detail,
        driver_detail=drv.detail,
        device_name=repr(device),
        backend=backend,
        tensor_type=tensor_type,
    )
