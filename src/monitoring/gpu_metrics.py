"""GPU metrics monitoring."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# try to import GPU monitoring libraries
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

try:
    from jtop import jtop

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False


@dataclass
class GPUMetrics:
    """Container for GPU metrics."""

    timestamp: str
    device_id: int
    name: str
    temperature_c: float
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None


class GPUMonitor:
    """Monitor GPU metrics using pynvml (NVIDIA) or jtop (Jetson).

    Automatically detects available GPU monitoring library.
    """

    def __init__(self):
        """Initialize GPU monitor."""
        self.backend = None
        self._jtop_instance = None
        self._metrics_history = []

        if PYNVML_AVAILABLE:
            self.backend = "pynvml"
            self.device_count = pynvml.nvmlDeviceGetCount()
        elif JTOP_AVAILABLE:
            self.backend = "jtop"
            self.device_count = 1  # Jetson has integrated GPU
        else:
            self.backend = None
            self.device_count = 0

    def is_available(self) -> bool:
        """Check if GPU monitoring is available.

        Returns:
            True if GPU monitoring is available
        """
        return self.backend is not None

    def get_metrics(self, device_id: int = 0) -> Optional[GPUMetrics]:
        """Get GPU metrics for a specific device.

        Args:
            device_id: GPU device ID

        Returns:
            GPUMetrics object or None if not available
        """
        if not self.is_available():
            return None

        if self.backend == "pynvml":
            return self._get_nvidia_metrics(device_id)
        elif self.backend == "jtop":
            return self._get_jetson_metrics()

        return None

    def _get_nvidia_metrics(self, device_id: int) -> Optional[GPUMetrics]:
        """Get metrics from NVIDIA GPU using pynvml."""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except pynvml.NVMLError:
                power_draw = None

            try:
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except pynvml.NVMLError:
                power_limit = None

            metrics = GPUMetrics(
                timestamp=datetime.now().isoformat(),
                device_id=device_id,
                name=name,
                temperature_c=float(temp),
                utilization_percent=float(util.gpu),
                memory_used_mb=mem_info.used / (1024**2),
                memory_total_mb=mem_info.total / (1024**2),
                memory_percent=(mem_info.used / mem_info.total) * 100,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
            )

            self._metrics_history.append(metrics)
            return metrics

        except pynvml.NVMLError as e:
            return None

    def _get_jetson_metrics(self) -> Optional[GPUMetrics]:
        """Get metrics from Jetson GPU using jtop."""
        try:
            with jtop() as jetson:
                gpu = jetson.gpu

                # jetson stats structure varies by model
                gpu_util = gpu.get("val", 0) if isinstance(gpu, dict) else 0

                stats = jetson.stats
                temp = stats.get("Temp GPU", 0)

                # memory info
                ram = jetson.ram
                mem_used = ram.get("use", 0) / 1024  # KB to MB
                mem_total = ram.get("tot", 1) / 1024

                # power
                power = jetson.power
                power_draw = power.get("tot", {}).get("power", 0) / 1000.0 if power else None

                metrics = GPUMetrics(
                    timestamp=datetime.now().isoformat(),
                    device_id=0,
                    name="Jetson GPU",
                    temperature_c=float(temp),
                    utilization_percent=float(gpu_util),
                    memory_used_mb=float(mem_used),
                    memory_total_mb=float(mem_total),
                    memory_percent=(mem_used / mem_total * 100) if mem_total > 0 else 0,
                    power_draw_w=power_draw,
                    power_limit_w=None,
                )

                self._metrics_history.append(metrics)
                return metrics

        except Exception as e:
            return None

    def get_all_metrics(self) -> List[GPUMetrics]:
        """Get metrics for all GPUs.

        Returns:
            List of GPUMetrics objects
        """
        metrics = []
        for i in range(self.device_count):
            m = self.get_metrics(i)
            if m:
                metrics.append(m)
        return metrics

    def get_metrics_dict(self, device_id: int = 0) -> Dict:
        """Get GPU metrics as dictionary.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary of metric values
        """
        metrics = self.get_metrics(device_id)
        if metrics is None:
            return {}

        return {
            "gpu_name": metrics.name,
            "gpu_temp_c": metrics.temperature_c,
            "gpu_util_percent": metrics.utilization_percent,
            "gpu_mem_used_mb": metrics.memory_used_mb,
            "gpu_mem_percent": metrics.memory_percent,
            "gpu_power_w": metrics.power_draw_w,
        }

    def get_history(self) -> list:
        """Get metrics history."""
        return self._metrics_history

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history = []

    def __del__(self):
        """Cleanup pynvml on destruction."""
        if PYNVML_AVAILABLE and self.backend == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except:
                pass
