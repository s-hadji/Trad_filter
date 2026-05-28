"""
GPU acceleration via cuDF.pandas (drop-in GPU replacement for pandas).

Usage
-----
    from gpu_support import gpu_init, GPU_STATE

    gpu_init()          # call once at startup
    print(GPU_STATE)    # dict with status info

If cuDF is available and CUDA is found:
    cudf.pandas.install() patches pandas globally →
    ALL existing pandas code (filter_engine, data_provider…) runs on GPU
    with zero other changes.
"""
from __future__ import annotations

from typing import Any, Dict

GPU_STATE: Dict[str, Any] = {
    "enabled":  False,
    "device":   "CPU only",
    "memory_gb": None,
    "compute":   None,
    "cudf":      False,
    "error":     None,
}


def gpu_init() -> bool:
    """
    Detect CUDA + cuDF, enable GPU pandas if possible.
    Returns True if GPU acceleration is active.
    Updates GPU_STATE in-place.
    """
    # ── Step 1: cuDF installed? ────────────────────────────────────────────
    try:
        import cudf  # noqa: F401
        GPU_STATE["cudf"] = True
    except ImportError:
        GPU_STATE["error"] = "cuDF not installed  (pip install cudf-cu12)"
        return False

    # ── Step 2: CUDA device reachable? ────────────────────────────────────
    try:
        import cupy
        n = cupy.cuda.runtime.getDeviceCount()
        if n == 0:
            GPU_STATE["error"] = "CUDA detected but no GPU device found"
            return False

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name  = props["name"]
        if isinstance(name, (bytes, bytearray)):
            name = name.decode("utf-8", errors="replace").strip("\x00")

        GPU_STATE["device"]    = name
        GPU_STATE["memory_gb"] = round(props["totalGlobalMem"] / 1e9, 1)
        GPU_STATE["compute"]   = f"{props['major']}.{props['minor']}"

    except ImportError:
        GPU_STATE["error"] = "cuPy not installed  (pip install cupy-cuda12x)"
        return False
    except Exception as e:
        GPU_STATE["error"] = f"CUDA check failed: {e}"
        return False

    # ── Step 3: Patch pandas globally with cuDF ────────────────────────────
    try:
        import cudf.pandas
        cudf.pandas.install()          # makes `import pandas` return cuDF on GPU
        GPU_STATE["enabled"] = True
        GPU_STATE["error"]   = None
        return True
    except Exception as e:
        GPU_STATE["error"] = f"cudf.pandas.install() failed: {e}"
        return False


def hardware_report() -> str:
    """Return a human-readable multi-line report of GPU state."""
    s = GPU_STATE
    lines = [
        "── GPU Acceleration ──────────────────────",
        f"  cuDF installed : {'✔' if s['cudf']    else '✘'}",
        f"  CUDA device    : {s['device']}",
    ]
    if s["memory_gb"] is not None:
        lines.append(f"  VRAM           : {s['memory_gb']} GB")
    if s["compute"] is not None:
        lines.append(f"  Compute cap.   : {s['compute']}")
    lines.append(f"  Status         : {'✔ GPU active (cuDF.pandas)' if s['enabled'] else '✘ CPU fallback'}")
    if s["error"]:
        lines.append(f"  Reason         : {s['error']}")
    lines.append("──────────────────────────────────────────")
    return "\n".join(lines)
