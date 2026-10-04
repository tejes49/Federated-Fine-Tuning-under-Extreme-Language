"""Environment and memory reporting. Reports only what can be measured; missing info is None."""
import os
import platform
import sys


def _ram_total_gb():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 ** 2, 2)   # kB -> GB
    except OSError:
        pass
    return None


def describe_environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "ram_total_gb": _ram_total_gb(),
           "torch": None, "cuda_available": False, "gpu_name": None, "gpu_memory_gb": None}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 2)
    except Exception:  # torch missing or broken install
        pass
    return env


def peak_memory_gb() -> dict:
    """Peak process RSS (all platforms with `resource`) and peak CUDA allocation (if a GPU is used)."""
    out = {"peak_process_rss_gb": None, "peak_cuda_allocated_gb": None}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # kB on Linux, bytes on macOS
        out["peak_process_rss_gb"] = round(rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 3)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_cuda_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    except Exception:
        pass
    return out
