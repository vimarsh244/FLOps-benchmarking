"""Power consumption monitoring."""

import os
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# try to import jtop for Jetson power monitoring
try:
    from jtop import jtop

    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False


@dataclass
class PowerMetrics:
    """Container for power metrics."""

    timestamp: str
    device_type: str
    total_power_w: Optional[float] = None
    cpu_power_w: Optional[float] = None
    gpu_power_w: Optional[float] = None
    soc_power_w: Optional[float] = None
    memory_power_w: Optional[float] = None


class PowerMonitor:
    """Monitor power consumption.

    Supports:
    - NVIDIA GPUs (via pynvml)
    - Jetson devices (via jtop)
    - Intel RAPL (on supported Linux systems)
    - Raspberry Pi (via vcgencmd)
    """

    def __init__(self, device_type: str = "auto"):
        """Initialize power monitor.

        Args:
            device_type: Device type (auto, nvidia, jetson, raspberry_pi, intel_rapl,
                raspberry_pi_4b, raspberry_pi_5, jetson_orin_nx)
        """
        if device_type == "auto":
            detected = self._detect_device()
            self.device_type_raw = detected
            self.device_type = self._normalize_device_type(detected)
        else:
            self.device_type_raw = device_type
            self.device_type = self._normalize_device_type(device_type)
        self._metrics_history = []
        self._last_rapl_values = {}

    @staticmethod
    def _normalize_device_type(device_type: str) -> str:
        """Normalize device type to a supported family.

        Args:
            device_type: Raw device type string

        Returns:
            Normalized device family string
        """
        if not device_type:
            return "unknown"

        value = device_type.strip().lower()

        if value.startswith("jetson"):
            return "jetson"
        if value in {"raspberry_pi", "raspberry_pi_4b", "raspberry_pi_5", "rpi4", "rpi5"}:
            return "raspberry_pi"
        if value in {"nvidia", "nvidia_gpu", "gpu"}:
            return "nvidia"
        if value in {"intel_rapl", "rapl", "intel"}:
            return "intel_rapl"
        if value == "auto":
            return "unknown"

        return value

    def _detect_device(self) -> str:
        """Auto-detect device type.

        Returns:
            Detected device type string
        """
        # check device model if available
        model_paths = [
            Path("/proc/device-tree/model"),
            Path("/sys/firmware/devicetree/base/model"),
        ]
        for model_path in model_paths:
            if model_path.exists():
                try:
                    with open(model_path) as f:
                        model = f.read().lower()
                    if "jetson" in model:
                        if "orin" in model and "nx" in model:
                            return "jetson_orin_nx"
                        return "jetson"
                    if "raspberry" in model:
                        if "raspberry pi 5" in model:
                            return "raspberry_pi_5"
                        if "raspberry pi 4" in model:
                            return "raspberry_pi_4b"
                        return "raspberry_pi"
                except Exception:
                    pass

        # check for jetson
        if Path("/etc/nv_tegra_release").exists() or JTOP_AVAILABLE:
            return "jetson"

        # check for raspberry pi
        if Path("/proc/device-tree/model").exists():
            try:
                with open("/proc/device-tree/model") as f:
                    if "raspberry" in f.read().lower():
                        return "raspberry_pi"
            except:
                pass

        # check for intel rapl
        if Path("/sys/class/powercap/intel-rapl").exists():
            return "intel_rapl"

        # check for nvidia gpu
        try:
            import pynvml

            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                pynvml.nvmlShutdown()
                return "nvidia"
        except:
            pass

        return "unknown"

    def is_available(self) -> bool:
        """Check if power monitoring is available.

        Returns:
            True if power monitoring is available
        """
        return self.device_type in {"jetson", "nvidia", "raspberry_pi", "intel_rapl"}

    def get_metrics(self) -> Optional[PowerMetrics]:
        """Get current power metrics.

        Returns:
            PowerMetrics object or None if not available
        """
        if self.device_type == "jetson":
            return self._get_jetson_power()
        elif self.device_type == "nvidia":
            return self._get_nvidia_power()
        elif self.device_type == "raspberry_pi":
            return self._get_rpi_power()
        elif self.device_type == "intel_rapl":
            return self._get_rapl_power()

        return None

    def _get_jetson_power(self) -> Optional[PowerMetrics]:
        """Get power metrics from Jetson device."""
        if not JTOP_AVAILABLE:
            return None

        try:
            with jtop() as jetson:
                power = jetson.power

                if not power:
                    return None

                total = power.get("tot", {})
                total_power = total.get("power", 0) / 1000.0  # mW to W

                # try to get component-level power
                cpu_power = None
                gpu_power = None
                soc_power = None

                for key, val in power.items():
                    if "cpu" in key.lower():
                        cpu_power = val.get("power", 0) / 1000.0
                    elif "gpu" in key.lower():
                        gpu_power = val.get("power", 0) / 1000.0
                    elif "soc" in key.lower():
                        soc_power = val.get("power", 0) / 1000.0

                metrics = PowerMetrics(
                    timestamp=datetime.now().isoformat(),
                    device_type=self.device_type_raw or "jetson",
                    total_power_w=total_power,
                    cpu_power_w=cpu_power,
                    gpu_power_w=gpu_power,
                    soc_power_w=soc_power,
                )

                self._metrics_history.append(metrics)
                return metrics

        except Exception as e:
            return None

    def _get_nvidia_power(self) -> Optional[PowerMetrics]:
        """Get power metrics from NVIDIA GPU."""
        try:
            import pynvml

            pynvml.nvmlInit()

            total_power = 0.0
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    total_power += power
                except pynvml.NVMLError:
                    pass

            pynvml.nvmlShutdown()

            metrics = PowerMetrics(
                timestamp=datetime.now().isoformat(),
                device_type=self.device_type_raw or "nvidia",
                total_power_w=total_power,
                gpu_power_w=total_power,
            )

            self._metrics_history.append(metrics)
            return metrics

        except Exception as e:
            return None

    def _get_rpi_power(self) -> Optional[PowerMetrics]:
        """Get power metrics from Raspberry Pi.

        Note: Raspberry Pi doesn't have direct power measurement.
        This returns voltage and estimates power based on it.
        """
        try:
            import subprocess

            # get voltage
            result = subprocess.run(
                ["vcgencmd", "measure_volts", "core"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                # parse voltage (format: volt=1.2000V)
                voltage_str = result.stdout.strip().split("=")[1].replace("V", "")
                voltage = float(voltage_str)

                # estimate power (rough approximation)
                # RPi 5 typically uses 3-5W idle, 6-10W under load
                estimated_power = voltage * 2.0  # rough estimate

                metrics = PowerMetrics(
                    timestamp=datetime.now().isoformat(),
                    device_type=self.device_type_raw or "raspberry_pi",
                    total_power_w=estimated_power,
                    soc_power_w=estimated_power,
                )

                self._metrics_history.append(metrics)
                return metrics

        except Exception as e:
            pass

        return None

    def _get_rapl_power(self) -> Optional[PowerMetrics]:
        """Get power metrics from Intel RAPL."""
        try:
            rapl_path = Path("/sys/class/powercap/intel-rapl")

            if not rapl_path.exists():
                return None

            total_power = 0.0
            cpu_power = 0.0

            # read package power
            for domain in rapl_path.iterdir():
                if domain.name.startswith("intel-rapl:"):
                    energy_path = domain / "energy_uj"
                    if energy_path.exists():
                        with open(energy_path) as f:
                            energy_uj = int(f.read().strip())

                        # calculate power from energy delta
                        key = str(domain)
                        if key in self._last_rapl_values:
                            last_energy, last_time = self._last_rapl_values[key]
                            time_delta = time.time() - last_time
                            if time_delta > 0:
                                energy_delta = energy_uj - last_energy
                                power_w = (energy_delta / 1e6) / time_delta
                                total_power += power_w

                                if "package" in domain.name:
                                    cpu_power = power_w

                        self._last_rapl_values[key] = (energy_uj, time.time())

            metrics = PowerMetrics(
                timestamp=datetime.now().isoformat(),
                device_type=self.device_type_raw or "intel_rapl",
                total_power_w=total_power if total_power > 0 else None,
                cpu_power_w=cpu_power if cpu_power > 0 else None,
            )

            self._metrics_history.append(metrics)
            return metrics

        except Exception as e:
            return None

    def get_metrics_dict(self) -> Dict:
        """Get power metrics as dictionary.

        Returns:
            Dictionary of metric values
        """
        metrics = self.get_metrics()
        if metrics is None:
            return {}

        return {
            "power_total_w": metrics.total_power_w,
            "power_cpu_w": metrics.cpu_power_w,
            "power_gpu_w": metrics.gpu_power_w,
            "power_device_type": metrics.device_type,
        }

    def get_history(self) -> list:
        """Get metrics history."""
        return self._metrics_history

    def clear_history(self) -> None:
        """Clear metrics history."""
        self._metrics_history = []
