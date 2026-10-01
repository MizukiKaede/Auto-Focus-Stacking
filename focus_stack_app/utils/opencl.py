"""OpenCL capability diagnostics. GPU operations require separate A/B approval."""
from __future__ import annotations

import logging


def opencl_status():
    status = dict(have_opencl=False, use_opencl=False, device=None, error=None,
                  enabled_operations=[], policy="cpu_until_measured_equivalence_and_speedup")
    try:
        import cv2
        status["have_opencl"] = bool(cv2.ocl.haveOpenCL())
        status["use_opencl"] = bool(cv2.ocl.useOpenCL())
        if status["use_opencl"]:
            device = cv2.ocl.Device_getDefault()
            status["device"] = dict(name=device.name(), vendor=device.vendorName(),
                                    version=device.version(), global_memory_bytes=int(device.globalMemSize()))
    except Exception as exc:
        status["error"] = str(exc)
    return status


def log_opencl_status(logger=None):
    status = opencl_status()
    (logger or logging.getLogger(__name__)).info("OpenCL runtime %s", status)
    return status
