"""Unified metrics collector combining all monitoring sources."""

import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from src.monitoring.system_metrics import SystemMonitor, PSUTIL_AVAILABLE
from src.monitoring.gpu_metrics import GPUMonitor
from src.monitoring.power_metrics import PowerMonitor


@dataclass
class CollectedMetrics:
    """Container for all collected metrics."""

    timestamp: str
    system: Optional[Dict] = None
    gpu: Optional[Dict] = None
    power: Optional[Dict] = None
    network: Optional[Dict] = None


class MetricsCollector:
    """Unified metrics collector combining system, GPU, and power monitoring.

    Supports:
    - Periodic background collection
    - On-demand collection
    - Multiple callback handlers
    """

    def __init__(
        self,
        collect_interval: float = 5.0,
        enable_system: bool = True,
        enable_gpu: bool = True,
        enable_power: bool = True,
    ):
        """Initialize metrics collector.

        Args:
            collect_interval: Interval between background collections (seconds)
            enable_system: Enable system metrics collection
            enable_gpu: Enable GPU metrics collection
            enable_power: Enable power metrics collection
        """
        self.collect_interval = collect_interval
        self.enable_system = enable_system
        self.enable_gpu = enable_gpu
        self.enable_power = enable_power

        # initialize monitors
        self._system_monitor = None
        self._gpu_monitor = None
        self._power_monitor = None

        if enable_system and PSUTIL_AVAILABLE:
            self._system_monitor = SystemMonitor()

        if enable_gpu:
            self._gpu_monitor = GPUMonitor()
            if not self._gpu_monitor.is_available():
                self._gpu_monitor = None

        if enable_power:
            self._power_monitor = PowerMonitor()
            if not self._power_monitor.is_available():
                self._power_monitor = None

        # collection state
        self._metrics_history: List[CollectedMetrics] = []
        self._callbacks: List[Callable[[CollectedMetrics], None]] = []
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None

    def is_available(self) -> Dict[str, bool]:
        """Check availability of each monitoring type.

        Returns:
            Dictionary of availability status
        """
        return {
            "system": self._system_monitor is not None,
            "gpu": self._gpu_monitor is not None,
            "power": self._power_monitor is not None,
        }

    def add_callback(self, callback: Callable[[CollectedMetrics], None]) -> None:
        """Add a callback to be called when metrics are collected.

        Args:
            callback: Function to call with collected metrics
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def collect(self) -> CollectedMetrics:
        """Collect metrics from all sources.

        Returns:
            CollectedMetrics object with all available metrics
        """
        timestamp = datetime.now().isoformat()

        # collect system metrics
        system_metrics = None
        if self._system_monitor:
            try:
                system_metrics = self._system_monitor.get_metrics_dict()
            except Exception:
                pass

        # collect gpu metrics
        gpu_metrics = None
        if self._gpu_monitor:
            try:
                gpu_metrics = self._gpu_monitor.get_metrics_dict()
            except Exception:
                pass

        # collect power metrics
        power_metrics = None
        if self._power_monitor:
            try:
                power_metrics = self._power_monitor.get_metrics_dict()
            except Exception:
                pass

        collected = CollectedMetrics(
            timestamp=timestamp,
            system=system_metrics,
            gpu=gpu_metrics,
            power=power_metrics,
        )

        self._metrics_history.append(collected)

        # call callbacks
        for callback in self._callbacks:
            try:
                callback(collected)
            except Exception:
                pass

        return collected

    def collect_flat(self) -> Dict:
        """Collect metrics and return as flat dictionary.

        Returns:
            Flat dictionary of all metrics
        """
        collected = self.collect()

        flat = {"timestamp": collected.timestamp}

        if collected.system:
            flat.update({f"sys_{k}": v for k, v in collected.system.items()})

        if collected.gpu:
            flat.update({f"gpu_{k}": v for k, v in collected.gpu.items()})

        if collected.power:
            flat.update({f"pwr_{k}": v for k, v in collected.power.items()})

        return flat

    def start_background_collection(self) -> None:
        """Start periodic background collection."""
        if self._collecting:
            return

        self._collecting = True
        self._collection_thread = threading.Thread(
            target=self._background_collection_loop,
            daemon=True,
        )
        self._collection_thread.start()

    def stop_background_collection(self) -> None:
        """Stop background collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
            self._collection_thread = None

    def _background_collection_loop(self) -> None:
        """Background collection loop."""
        while self._collecting:
            try:
                self.collect()
            except Exception:
                pass
            time.sleep(self.collect_interval)

    def get_history(self) -> List[CollectedMetrics]:
        """Get collected metrics history.

        Returns:
            List of CollectedMetrics objects
        """
        return self._metrics_history

    def get_history_flat(self) -> List[Dict]:
        """Get collected metrics history as flat dictionaries.

        Returns:
            List of flat metric dictionaries
        """
        result = []
        for m in self._metrics_history:
            flat = {"timestamp": m.timestamp}
            if m.system:
                flat.update({f"sys_{k}": v for k, v in m.system.items()})
            if m.gpu:
                flat.update({f"gpu_{k}": v for k, v in m.gpu.items()})
            if m.power:
                flat.update({f"pwr_{k}": v for k, v in m.power.items()})
            result.append(flat)
        return result

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history = []
        if self._system_monitor:
            self._system_monitor.clear_history()
        if self._gpu_monitor:
            self._gpu_monitor.clear_history()
        if self._power_monitor:
            self._power_monitor.clear_history()

    def get_summary(self) -> Dict:
        """Get summary statistics of collected metrics.

        Returns:
            Dictionary of summary statistics
        """
        if not self._metrics_history:
            return {}

        import statistics

        def calc_stats(values: List[float]) -> Dict:
            if not values:
                return {}
            return {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }

        summary = {}

        # system metrics summary
        cpu_values = []
        mem_values = []
        for m in self._metrics_history:
            if m.system:
                if "cpu_percent" in m.system:
                    cpu_values.append(m.system["cpu_percent"])
                if "memory_percent" in m.system:
                    mem_values.append(m.system["memory_percent"])

        if cpu_values:
            summary["cpu_percent"] = calc_stats(cpu_values)
        if mem_values:
            summary["memory_percent"] = calc_stats(mem_values)

        # gpu metrics summary
        gpu_util_values = []
        gpu_mem_values = []
        for m in self._metrics_history:
            if m.gpu:
                if "gpu_util_percent" in m.gpu:
                    gpu_util_values.append(m.gpu["gpu_util_percent"])
                if "gpu_mem_percent" in m.gpu:
                    gpu_mem_values.append(m.gpu["gpu_mem_percent"])

        if gpu_util_values:
            summary["gpu_util_percent"] = calc_stats(gpu_util_values)
        if gpu_mem_values:
            summary["gpu_mem_percent"] = calc_stats(gpu_mem_values)

        # power metrics summary
        power_values = []
        for m in self._metrics_history:
            if m.power and m.power.get("power_total_w"):
                power_values.append(m.power["power_total_w"])

        if power_values:
            summary["power_total_w"] = calc_stats(power_values)
            summary["energy_total_wh"] = sum(power_values) * self.collect_interval / 3600

        return summary

    def __enter__(self):
        self.start_background_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_background_collection()
        return False
