"""Amostragem periódica de GPU (via NVML) e de RAM do processo do llama-server."""

import csv
import threading
import time

import psutil
import pynvml

MB = 1024 * 1024


class GPU:
    def __init__(self, index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def info(self):
        return {
            "name": pynvml.nvmlDeviceGetName(self.handle),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "vram_total_mb": pynvml.nvmlDeviceGetMemoryInfo(self.handle).total / MB,
        }

    def snapshot(self):
        h = self.handle
        return {
            "vram_used_mb": round(pynvml.nvmlDeviceGetMemoryInfo(h).used / MB, 1),
            "gpu_util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
            "gpu_temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "gpu_sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "gpu_power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
        }


class ResourceMonitor:
    """Thread que registra uma amostra a cada `interval` segundos.

    As amostras recebem o rótulo da fase corrente (loading, idle, battery),
    permitindo separar VRAM pós-carga de VRAM de pico durante a bateria.
    """

    def __init__(self, gpu, interval=0.25):
        self.gpu = gpu
        self.interval = interval
        self.samples = []
        self.phase = "loading"
        self._proc = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = None

    def start(self, pid):
        self._proc = psutil.Process(pid)
        self._t0 = time.perf_counter()
        self._thread.start()

    def set_phase(self, phase):
        self.phase = phase

    def _ram_mb(self):
        try:
            return round(self._proc.memory_info().rss / MB, 1)
        except psutil.Error:
            return None

    def _run(self):
        while not self._stop.is_set():
            sample = {"t_s": round(time.perf_counter() - self._t0, 3), "phase": self.phase}
            sample.update(self.gpu.snapshot())
            sample["ram_rss_mb"] = self._ram_mb()
            self.samples.append(sample)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self._thread.join()

    def save_csv(self, path):
        if not self.samples:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.samples[0]))
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self):
        def values(key, phase=None, busy=False):
            return [s[key] for s in self.samples
                    if s[key] is not None
                    and (phase is None or s["phase"] == phase)
                    and (not busy or s["gpu_util_pct"] >= 50)]

        # Clock mínimo só com a GPU ocupada: entre requisições o clock cai por
        # ociosidade, o que não indica limitação térmica.
        busy_clock = values("gpu_sm_clock_mhz", "battery", busy=True)
        return {
            "vram_loaded_mb": _max(values("vram_used_mb", "idle")),
            "vram_peak_mb": _max(values("vram_used_mb")),
            "ram_peak_mb": _max(values("ram_rss_mb")),
            "gpu_temp_max_c": _max(values("gpu_temp_c", "battery")),
            "gpu_sm_clock_min_busy_mhz": min(busy_clock) if busy_clock else None,
            "gpu_sm_clock_avg_busy_mhz": _mean(busy_clock),
            "gpu_power_avg_w": _mean(values("gpu_power_w", "battery")),
            "gpu_util_avg_pct": _mean(values("gpu_util_pct", "battery")),
            "monitor_samples": len(self.samples),
        }


def _max(xs):
    return max(xs) if xs else None


def _mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else None
