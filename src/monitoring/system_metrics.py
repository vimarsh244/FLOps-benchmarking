"""System metrics monitoring (CPU, memory, network)."""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class SystemMetrics:
    """Container for system metrics."""

    timestamp: str
    cpu_percent: float
    cpu_per_core: list
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_read_mb: float
    disk_write_mb: float
    net_bytes_sent_mb: float
    net_bytes_recv_mb: float


class SystemMonitor:
    """Monitor system resources (CPU, memory, disk, network).

    Uses psutil to collect system metrics.
    """

    def __init__(self, interval: float = 1.0):
        """Initialize system monitor.

        Args:
            interval: Minimum interval between measurements (seconds)
        """
        if not PSUTIL_AVAILABLE:
            raise ImportError(
                "psutil is required for system monitoring. Install with: pip install psutil"
            )

        self.interval = interval
        self._last_measure_time = 0.0
        self._last_disk_io = None
        self._last_net_io = None
        self._metrics_history = []

    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics.

        Returns:
            SystemMetrics object with current values
        """
        # get cpu metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

        # get memory metrics
        mem = psutil.virtual_memory()
        memory_percent = mem.percent
        memory_used_gb = mem.used / (1024**3)
        memory_available_gb = mem.available / (1024**3)

        # get disk io metrics (delta since last call)
        disk_io = psutil.disk_io_counters()
        if self._last_disk_io is None:
            disk_read_mb = 0.0
            disk_write_mb = 0.0
        else:
            disk_read_mb = (disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024**2)
            disk_write_mb = (disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024**2)
        self._last_disk_io = disk_io

        # get network io metrics (delta since last call)
        net_io = psutil.net_io_counters()
        if self._last_net_io is None:
            net_bytes_sent_mb = 0.0
            net_bytes_recv_mb = 0.0
        else:
            net_bytes_sent_mb = (net_io.bytes_sent - self._last_net_io.bytes_sent) / (1024**2)
            net_bytes_recv_mb = (net_io.bytes_recv - self._last_net_io.bytes_recv) / (1024**2)
        self._last_net_io = net_io

        metrics = SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            net_bytes_sent_mb=net_bytes_sent_mb,
            net_bytes_recv_mb=net_bytes_recv_mb,
        )

        self._metrics_history.append(metrics)
        self._last_measure_time = time.time()

        return metrics

    def get_metrics_dict(self) -> Dict:
        """Get current system metrics as dictionary.

        Returns:
            Dictionary of metric values
        """
        metrics = self.get_metrics()
        return {
            "timestamp": metrics.timestamp,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_gb": metrics.memory_used_gb,
            "net_sent_mb": metrics.net_bytes_sent_mb,
            "net_recv_mb": metrics.net_bytes_recv_mb,
        }

    def get_history(self) -> list:
        """Get metrics history.

        Returns:
            List of SystemMetrics objects
        """
        return self._metrics_history

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history = []

    @staticmethod
    def get_system_info() -> Dict:
        """Get static system information.

        Returns:
            Dictionary of system information
        """
        if not PSUTIL_AVAILABLE:
            return {}

        return {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": {
                "system": __import__("platform").system(),
                "release": __import__("platform").release(),
                "machine": __import__("platform").machine(),
            },
        }
